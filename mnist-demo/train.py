import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_data_loaders
from models import build_model
from models.ops_quant import QASSGD


def parse_args() -> argparse.Namespace:
    # 训练脚本只负责“跑一次训练并落盘”。
    # 对比不同方法时，不在这里混入评估和汇总逻辑，而是分别交给 evaluate.py / compare.py。
    parser = argparse.ArgumentParser(description="Train one MNIST demo run")
    parser.add_argument("--mode", choices=["float", "quant", "quant_qas"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default="mnist-demo/artifacts/data")
    parser.add_argument("--output-root", default="mnist-demo/artifacts/runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--float-lr", type=float, default=0.05)
    parser.add_argument("--quant-lr", type=float, default=0.05)
    parser.add_argument("--qas-lr", type=float, default=0.02)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--a-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_optimizer(mode: str, model: nn.Module, args: argparse.Namespace) -> tuple[torch.optim.Optimizer, float]:
    # 1. float: 浮点模型 + 标准 SGD
    # 2. quant: 量化训练模型 + 标准 SGD
    # 3. quant_qas: 量化训练模型 + QASSGD
    #
    # QASSGD 本质上仍然是 SGD，只是在真正 step 之前多做一次 pre_step(model)，
    # 用各层记录下来的量化 scale 去重标定梯度。
    if mode == "float":
        lr = args.float_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    elif mode == "quant":
        lr = args.quant_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    elif mode == "quant_qas":
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
        # 1. 清空上一步残留梯度
        # 2. 前向传播
        # 3. 计算 loss
        # 4. 反向传播得到 grad
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # QAS 和普通 SGD 的分叉点就在这里。
        #
        # 对 quant 模式：
        #   optimizer 是 torch.optim.SGD，没有 pre_step，梯度会直接被 step() 使用。
        #
        # 对 quant_qas 模式：
        #   optimizer 是 QASSGD，会先执行 pre_step(model)。
        #   pre_step 会遍历模型中的 QuantizedLinear，
        #   读取这些层在前向里缓存下来的 input / weight / output scale，
        #   再用这些 scale 去修正当前 batch 的 weight.grad 和 bias.grad。
        #
        # 也就是说，这个 demo 里 QAS 不改前向公式，也不改 loss，
        # 它只改“反向之后、参数更新之前”这一小步。
        if hasattr(optimizer, "pre_step"):
            optimizer.pre_step(model)

        # 真正的参数更新仍然走标准的 optimizer.step()。
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
    model = build_model(args.mode, w_bits=args.w_bits, a_bits=args.a_bits)
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
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "num_workers": args.num_workers,
        "momentum": args.momentum,
        "float_lr": args.float_lr,
        "quant_lr": args.quant_lr,
        "qas_lr": args.qas_lr,
        "w_bits": args.w_bits,
        "a_bits": args.a_bits,
        "history": history,
    }
    save_json(run_dir / "train_result.json", train_result)
    save_checkpoint(run_dir / "checkpoint.pt", {
        "mode": args.mode,
        "model_state_dict": model.state_dict(),
        "train_config": train_result,
    })
    print(f"Saved training artifacts to {run_dir}")


if __name__ == "__main__":
    main()
