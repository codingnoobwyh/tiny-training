import argparse
import json
from pathlib import Path

import torch
from torch import nn

from datasets import build_data_loaders
from models.mnist import FloatMNISTNet, build_native_ptq_prepare_model_from_float_state_dict


def parse_args() -> argparse.Namespace:
    # 把一个已经训练好的 FP32 checkpoint 做成 PTQ 模型。
    # 流程是：
    # 1. 读取 float checkpoint
    # 2. 构造原生 PTQ prepare 模型
    # 3. 加载浮点权重
    # 4. 用少量训练数据做校准
    # 5. convert 成量化推理模型
    # 6. 在测试集上评估并保存结果
    parser = argparse.ArgumentParser(description="Build PTQ model from a float MNIST checkpoint")
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default="mnist-demo/artifacts/data")
    parser.add_argument("--output-root", default="mnist-demo/artifacts/runs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--calibration-batches", type=int, default=32)
    return parser.parse_args()


def get_run_dir(output_root: str, run_name: str) -> Path:
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def save_checkpoint(path: Path, payload: dict) -> None:
    torch.save(payload, path)


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def calibrate_model(model: nn.Module, data_loader, calibration_batches: int) -> None:
    # PTQ 的关键步骤是校准。
    # observer 会在前向过程中统计激活分布，再据此确定 scale / zero_point。
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            model(images)
            if batch_idx + 1 >= calibration_batches:
                break


def evaluate_model(model: nn.Module, data_loader, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return {
        "loss": total_loss / total_samples,
        "top1": 100.0 * total_correct / total_samples,
    }


def extract_float_state_dict(checkpoint: dict) -> dict:
    # PTQ 只能从 float checkpoint 起步。
    if checkpoint["mode"] != "float":
        raise ValueError(f"PTQ expects a float checkpoint, got mode={checkpoint['mode']}")
    return checkpoint["model_state_dict"]


def main() -> None:
    args = parse_args()
    run_dir = get_run_dir(args.output_root, args.run_name)

    source_checkpoint = load_checkpoint(args.init_from)
    float_state_dict = extract_float_state_dict(source_checkpoint)

    # 这里先加载一份纯浮点模型，只是为了严格检查输入 checkpoint 的权重形状是否匹配。
    float_model = FloatMNISTNet()
    float_model.load_state_dict(float_state_dict, strict=True)

    train_loader, test_loader = build_data_loaders(
        args.data_root,
        args.batch_size,
        args.test_batch_size,
        args.num_workers,
    )

    # 先把浮点权重加载进 quantizable 母体，再做 fuse/prepare。
    # 这样卷积和全连接层的参数不会因为 fuse 后键名变化而丢失。
    prepared_model = build_native_ptq_prepare_model_from_float_state_dict(float_state_dict)

    # 用少量训练集数据做校准，observer 会在这里统计激活范围。
    calibrate_model(prepared_model, train_loader, args.calibration_batches)

    # convert 之后得到真正的量化推理模型。
    quantized_model = torch.ao.quantization.convert(prepared_model.eval(), inplace=False)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(quantized_model, test_loader, criterion)

    ptq_result = {
        "run_name": args.run_name,
        "mode": "ptq",
        "source_checkpoint": args.init_from,
        "calibration_batches": args.calibration_batches,
        "test_top1": metrics["top1"],
        "test_loss": metrics["loss"],
    }

    # 保存 compare.py 可直接读取的结果文件。
    save_json(run_dir / "eval_result.json", ptq_result)
    save_json(run_dir / "ptq_result.json", ptq_result)

    # 保存 PTQ 后的量化模型 checkpoint，后面 Route C 可以拿它当起点。
    save_checkpoint(run_dir / "checkpoint.pt", {
        "mode": "ptq",
        "model_state_dict": quantized_model.state_dict(),
        "source_checkpoint": args.init_from,
        "ptq_config": {
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "num_workers": args.num_workers,
            "calibration_batches": args.calibration_batches,
        },
    })

    print(
        f"{args.run_name}: mode=ptq "
        f"test_top1={metrics['top1']:.2f} test_loss={metrics['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
