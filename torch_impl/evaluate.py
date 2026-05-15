import argparse
import sys
from pathlib import Path

from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import DEFAULT_DATA_ROOT, build_data_loaders
from common.train_utils import evaluate_model, get_run_dir, load_checkpoint, save_json
from common.baseline import FloatMNISTNet, build_ptq_model, build_qat_model
from torch_impl.quantized.models import (
    QuantizedMNISTNet,
    initialize_quantized_model_from_ptq_checkpoint,
)


TORCH_IMPL_DIR = Path(__file__).resolve().parent
ROOT_DIR = TORCH_IMPL_DIR.parent
DEFAULT_OUTPUT_ROOT = TORCH_IMPL_DIR / "artifacts" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one trained run")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--as-quantized-forward", action="store_true")
    return parser.parse_args()


def load_float_run(checkpoint: dict) -> tuple[nn.Module, dict, str]:
    model = FloatMNISTNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["train_config"], checkpoint["mode"]


def load_qat_run(checkpoint: dict) -> tuple[nn.Module, dict, str]:
    model = build_qat_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["train_config"], "qat"


def load_ptq_run(checkpoint: dict) -> tuple[nn.Module, dict, str]:
    model = build_ptq_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["ptq_config"], "ptq"


def load_quantized_run(checkpoint: dict) -> tuple[nn.Module, dict, str]:
    model = QuantizedMNISTNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["train_config"], checkpoint["mode"]


def load_quantized_forward_from_ptq(checkpoint: dict) -> tuple[nn.Module, dict, str]:
    model = QuantizedMNISTNet()
    initialize_quantized_model_from_ptq_checkpoint(model, checkpoint)
    return model, checkpoint["ptq_config"], "quantized_forward_from_ptq"


def main() -> None:
    args = parse_args()
    run_dir = get_run_dir(args.output_root, args.run_name)
    checkpoint = load_checkpoint(run_dir / "checkpoint.pt")
    checkpoint_mode = checkpoint["mode"]

    if args.as_quantized_forward:
        if checkpoint_mode != "ptq":
            raise ValueError("--as-quantized-forward 只能用于 ptq run")
        model, data_config, result_mode = load_quantized_forward_from_ptq(checkpoint)
        result_path = run_dir / "eval_result_quantized_forward.json"
    elif checkpoint_mode == "float":
        model, data_config, result_mode = load_float_run(checkpoint)
        result_path = run_dir / "eval_result.json"
    elif checkpoint_mode == "qat":
        model, data_config, result_mode = load_qat_run(checkpoint)
        result_path = run_dir / "eval_result.json"
    elif checkpoint_mode == "ptq":
        model, data_config, result_mode = load_ptq_run(checkpoint)
        result_path = run_dir / "eval_result.json"
    elif checkpoint_mode in {"quantized", "qas"}:
        model, data_config, result_mode = load_quantized_run(checkpoint)
        result_path = run_dir / "eval_result.json"
    else:
        raise ValueError(f"Unsupported checkpoint mode: {checkpoint_mode}")

    _, test_loader = build_data_loaders(
        args.data_root,
        data_config["batch_size"],
        data_config["test_batch_size"],
        data_config["num_workers"],
    )
    metrics = evaluate_model(model, test_loader, nn.CrossEntropyLoss())
    eval_result = {
        "run_name": args.run_name,
        "mode": result_mode,
        "test_top1": metrics["top1"],
        "test_loss": metrics["loss"],
    }
    save_json(result_path, eval_result)
    print(f"{args.run_name}: mode={result_mode} test_top1={metrics['top1']:.2f} test_loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
