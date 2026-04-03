import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import build_data_loaders
from models import build_model, convert_qat_model_for_inference, initialize_quantized_model_from_ptq_checkpoint


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = DEMO_DIR / "artifacts" / "data"
DEFAULT_OUTPUT_ROOT = DEMO_DIR / "artifacts" / "runs"


def parse_args() -> argparse.Namespace:
    # 评估脚本只负责“读取一个已经训练好的 run，然后在测试集上算结果”。
    # 它不参与训练，也不负责多实验对比。
    parser = argparse.ArgumentParser(description="Evaluate one trained MNIST demo run")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--converted", action="store_true")
    parser.add_argument("--as-quantized-forward", action="store_true")
    return parser.parse_args()


def get_run_dir(output_root: str, run_name: str) -> Path:
    # 评估阶段沿用训练阶段的目录约定，直接去同名 run 目录里找 checkpoint。
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_checkpoint(path: Path) -> dict:
    # 评估完全依赖训练阶段保存的 checkpoint 恢复模型。
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, payload: dict) -> None:
    # 评估结果保存成独立 JSON，方便 compare.py 后续聚合。
    path.write_text(json.dumps(payload, indent=2))


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
) -> dict[str, float]:
    # 评估阶段切换到 eval()，关闭训练态行为。
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 评估时不需要梯度，关闭 autograd
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


def main() -> None:
    args = parse_args()
    run_dir = get_run_dir(args.output_root, args.run_name)

    # 评估完全基于训练阶段落盘的 checkpoint
    checkpoint = load_checkpoint(run_dir / "checkpoint.pt")
    checkpoint_mode = checkpoint["mode"]

    # 这个入口专门用于排查“PTQ -> 自定义 quantized 前向映射”是否正确。
    # 打开后：
    # 1. 输入仍然是一个 ptq checkpoint
    # 2. 但评估时不直接用原生 PTQ 模型
    # 3. 而是先构造我们自己的 quantized 模型，再把 PTQ 参数映射进去
    if args.as_quantized_forward:
        if checkpoint_mode != "ptq":
            raise ValueError("--as-quantized-forward 只能用于评估 ptq checkpoint")
        model = build_model("quantized")
        initialize_quantized_model_from_ptq_checkpoint(model, checkpoint)
        data_config = checkpoint["ptq_config"]
        result_mode = "quantized_forward_from_ptq"
        result_path = run_dir / "eval_result_quantized_forward.json"
    else:
        model = build_model(checkpoint_mode)
        model.load_state_dict(checkpoint["model_state_dict"])
        data_config = checkpoint["train_config"]
        result_mode = checkpoint_mode
        result_path = run_dir / "eval_result.json"

    if args.converted and checkpoint_mode == "qat":
        model = convert_qat_model_for_inference(model)
        result_mode = f"{checkpoint_mode}_converted"
        result_path = run_dir / "eval_result_converted.json"
    criterion = nn.CrossEntropyLoss()

    _, test_loader = build_data_loaders(
        args.data_root,
        data_config["batch_size"],
        data_config["test_batch_size"],
        data_config["num_workers"],
    )
    metrics = evaluate_model(model, test_loader, criterion)

    # eval_result*.json 只保留最终验收时真正关心的指标。
    # 更细的训练过程指标已经在 train_result.json 里。
    eval_result = {
        "run_name": args.run_name,
        "mode": result_mode,
        "test_top1": metrics["top1"],
        "test_loss": metrics["loss"],
    }
    save_json(result_path, eval_result)
    print(
        f"{args.run_name}: mode={result_mode} "
        f"test_top1={metrics['top1']:.2f} test_loss={metrics['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
