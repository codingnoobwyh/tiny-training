import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_data_loaders
from models import QASSGD, build_model, initialize_quantized_model_from_ptq_checkpoint


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = DEMO_DIR / "artifacts" / "data"
DEFAULT_OUTPUT_ROOT = DEMO_DIR / "artifacts" / "runs"


def parse_args() -> argparse.Namespace:
    # 训练脚本只负责“跑一次训练并落盘”。
    # 对比不同方法时，不在这里混入评估和汇总逻辑，而是分别交给 evaluate.py / compare.py。
    parser = argparse.ArgumentParser(description="Train one MNIST demo run")
    parser.add_argument("--mode", choices=["float", "qat", "quantized", "qas"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--float-lr", type=float, default=0.05)
    parser.add_argument("--qat-lr", type=float, default=0.05)
    parser.add_argument("--quantized-lr", type=float, default=0.01)
    parser.add_argument("--qas-lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_optimizer(mode: str, model: nn.Module, args: argparse.Namespace) -> tuple[torch.optim.Optimizer, float]:
    # 当前 demo 只保留两种训练模式：
    # 1. float: 标准浮点训练
    # 2. qat: 标准 QAT，前向里 quant -> dequant，再继续浮点训练
    # 3. quantized: 真实量化语义训练，不再走 fake quant
    # 4. qas: 在真实量化语义训练基础上，再加入 QAS 的梯度 scale 校正
    #
    # 两者在优化器层面都使用普通 SGD。
    # QAT 和 float 的区别不在 optimizer，而在模型前向是否插入 fake quant。
    if mode == "float":
        lr = args.float_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    elif mode == "qat":
        lr = args.qat_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    elif mode == "quantized":
        lr = args.quantized_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    elif mode == "qas":
        lr = args.qas_lr
        optimizer = QASSGD(model.parameters(), lr=lr, momentum=args.momentum)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return optimizer, lr


def get_run_dir(output_root: str, run_name: str) -> Path:
    # 每次训练产物都收在 output_root/run_name 下。
    # 这样一个实验的 checkpoint 和 json 结果会自然聚在一起。
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict) -> None:
    # 训练指标保存成 JSON，方便直接打开文件查看。
    path.write_text(json.dumps(payload, indent=2))


def save_checkpoint(path: Path, payload: dict) -> None:
    # checkpoint 除了模型权重，还保存重建模型所需的训练配置。
    torch.save(payload, path)


def load_checkpoint(path: str) -> dict:
    # 继续训练时统一从这里读取 checkpoint。
    return torch.load(path, map_location="cpu", weights_only=False)


def initialize_model_from_checkpoint(model: nn.Module, init_from: str, target_mode: str) -> dict:
    # 这个函数负责把“统一的 FP32 起点”分发到不同训练路线：
    #
    # Route A: float <- float checkpoint
    #   直接严格加载，同一个模型结构继续训练。
    #
    # Route B: qat <- float checkpoint
    #   只把共享的卷积/全连接权重拷进原生 QAT 模型，
    #   observer / fake quant 的状态仍然保持初始值，让 QAT 自己再校准。
    checkpoint = load_checkpoint(init_from)
    source_mode = checkpoint["mode"]
    source_state_dict = checkpoint["model_state_dict"]

    if target_mode == source_mode:
        model.load_state_dict(source_state_dict, strict=True)
        return checkpoint

    if target_mode == "qat" and source_mode == "float":
        target_state_dict = model.state_dict()
        shared_state_dict = {
            key: value
            for key, value in source_state_dict.items()
            if key in target_state_dict and target_state_dict[key].shape == value.shape
        }
        # 这里只加载共享的浮点权重参数，例如 conv1.weight / fc1.bias。
        # QAT 特有的 observer、scale、zero_point 状态不从 float checkpoint 恢复。
        missing, unexpected = model.load_state_dict(shared_state_dict, strict=False)
        print(
            "Initialized QAT model from float checkpoint: "
            f"loaded={len(shared_state_dict)} missing={len(missing)} unexpected={len(unexpected)}"
        )
        return checkpoint

    if target_mode in {"quantized", "qas"} and source_mode == "ptq":
        initialize_quantized_model_from_ptq_checkpoint(model, checkpoint)
        return checkpoint

    raise ValueError(
        f"Unsupported initialization path: target_mode={target_mode}, source_mode={source_mode}"
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    desc: str,
) -> dict[str, float]:
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(data_loader, desc=desc, leave=False)
    for images, labels in progress:
        # 标准 PyTorch 训练顺序：
        # 1. 清空梯度
        # 2. 前向传播
        # 3. 计算损失
        # 4. 反向传播
        # 5. SGD 更新
        #
        # 对 float 模式，前向完全是普通浮点网络。
        # 对 qat 模式，差别只在 model(images) 内部：
        # prepare_qat 注入的 observer/fake quant 会在前向中自动生效。
        # 对 quantized / qas 模式，前向已经不再是 fake quant，
        # 而是我们自定义的“整数码值语义 + effective_scale 重定标”。
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        # QAS 只在这里和普通 real quantized training 分叉：
        # backward 之后、SGD step 之前，按量化 scale 重新标定梯度。
        if hasattr(optimizer, "pre_step"):
            optimizer.pre_step(model)
        optimizer.step()

        # 训练时实时累计 loss 和 top1，最后写入 train_result.json。
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)

        progress.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "top1": f"{100.0 * total_correct / total_samples:.2f}",
        })

    return {
        "loss": total_loss / total_samples,
        "top1": 100.0 * total_correct / total_samples,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_loader, _ = build_data_loaders(
        args.data_root,
        args.batch_size,
        args.test_batch_size,
        args.num_workers,
    )
    model = build_model(args.mode)
    init_meta = None
    if args.init_from is not None:
        init_meta = initialize_model_from_checkpoint(model, args.init_from, args.mode)
    criterion = nn.CrossEntropyLoss()
    optimizer, lr = build_optimizer(args.mode, model, args)
    run_dir = get_run_dir(args.output_root, args.run_name)

    print(f"===== Training mode={args.mode}, lr={lr}, run_dir={run_dir} =====")

    history = []
    for epoch in range(args.epochs):
        # 每个 epoch 保存一次训练指标。
        # 这样后面即使不看终端输出，也能从 train_result.json 回看训练曲线。
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            desc=f"{args.mode} train {epoch + 1}/{args.epochs}",
        )
        history.append({
            "epoch": epoch,
            "train": train_metrics,
        })
        print(
            f"{args.mode} epoch={epoch} "
            f"train_top1={train_metrics['top1']:.2f} "
            f"train_loss={train_metrics['loss']:.4f}"
        )

    # 保存重建模型所需的信息
    train_result = {
        "mode": args.mode,
        "run_name": args.run_name,
        "init_from": args.init_from,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "num_workers": args.num_workers,
        "momentum": args.momentum,
        "float_lr": args.float_lr,
        "qat_lr": args.qat_lr,
        "quantized_lr": args.quantized_lr,
        "qas_lr": args.qas_lr,
        "history": history,
    }
    if init_meta is not None:
        train_result["init_from_mode"] = init_meta["mode"]
    save_json(run_dir / "train_result.json", train_result)
    save_checkpoint(run_dir / "checkpoint.pt", {
        "mode": args.mode,
        "model_state_dict": model.state_dict(),
        "train_config": train_result,
    })
    print(f"Saved training artifacts to {run_dir}")


if __name__ == "__main__":
    main()
