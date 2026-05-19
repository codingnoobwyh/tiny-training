import argparse
import json
from pathlib import Path

from common.constants import DEFAULT_NPY_QAS_RUN_ROOT, DEFAULT_QAS_RUN_NAME, DEFAULT_TORCH_QAS_RUN_ROOT

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TORCH_RUN_ROOT = ROOT_DIR / DEFAULT_TORCH_QAS_RUN_ROOT
DEFAULT_NPY_RUN_ROOT = ROOT_DIR / DEFAULT_NPY_QAS_RUN_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare torch QAS and npy QAS eval results")
    parser.add_argument("--torch-run", default=DEFAULT_QAS_RUN_NAME)
    parser.add_argument("--npy-run", default=DEFAULT_QAS_RUN_NAME)
    parser.add_argument("--torch-root", default=str(DEFAULT_TORCH_RUN_ROOT))
    parser.add_argument("--npy-root", default=str(DEFAULT_NPY_RUN_ROOT))
    return parser.parse_args()


def _load_eval_result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval result: {path}")
    return json.loads(path.read_text())


def _format_row(name: str, top1: float | str, loss: float | str) -> str:
    return f"{name:<10}  {top1:>10}  {loss:>10}"


def compare_qas(
    torch_run: str = DEFAULT_QAS_RUN_NAME,
    npy_run: str = DEFAULT_QAS_RUN_NAME,
    torch_root: str | Path = DEFAULT_TORCH_RUN_ROOT,
    npy_root: str | Path = DEFAULT_NPY_RUN_ROOT,
) -> dict:
    torch_result = _load_eval_result(Path(torch_root) / torch_run / "eval_result.json")
    npy_result = _load_eval_result(Path(npy_root) / npy_run / "eval_result.json")

    return {
        "torch": torch_result,
        "npy": npy_result,
        "top1_diff": npy_result["test_top1"] - torch_result["test_top1"],
        "loss_diff": npy_result["test_loss"] - torch_result["test_loss"],
    }


def main() -> None:
    args = parse_args()
    result = compare_qas(args.torch_run, args.npy_run, args.torch_root, args.npy_root)
    torch_result = result["torch"]
    npy_result = result["npy"]

    print(_format_row("impl", "test_top1", "test_loss"))
    print(_format_row("torch", f"{torch_result['test_top1']:.2f}", f"{torch_result['test_loss']:.4f}"))
    print(_format_row("npy", f"{npy_result['test_top1']:.2f}", f"{npy_result['test_loss']:.4f}"))
    print("-" * 34)
    print(_format_row("npy-torch", f"{result['top1_diff']:+.2f}", f"{result['loss_diff']:+.4f}"))


if __name__ == "__main__":
    main()
