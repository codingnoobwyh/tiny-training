import argparse
import json
from pathlib import Path


TORCH_IMPL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = TORCH_IMPL_DIR / "artifacts" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluated runs")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-names", nargs="+", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def format_row(type_label: str, mode_label: str, top1: float | str, loss: float | str) -> str:
    return f"{type_label:<6}  {mode_label:<8}  {top1:>9}  {loss:>9}"


def normalize_display_fields(row: dict) -> tuple[str, str]:
    run_name = row["run_name"]
    raw_mode = row["mode"]

    if run_name == "fp-base":
        return "fp32", "fp_base"
    if raw_mode == "ptq":
        return "int8", "ptq_base"
    if raw_mode == "float":
        return "fp32", "fp"
    if raw_mode == "qat":
        return "int8", "qat"
    if raw_mode == "qas":
        return "int8", "qas"
    if raw_mode in {"quantized", "quantized_forward_from_ptq"}:
        return "int8", "quant"
    return "int8", raw_mode


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = []

    for run_name in args.run_names:
        eval_path = output_root / run_name / "eval_result.json"
        result = load_json(eval_path)
        dtype_label, mode_label = normalize_display_fields(result)
        result["display_type"] = dtype_label
        result["display_mode"] = mode_label
        rows.append(result)

    baseline_modes = {"fp_base", "ptq_base"}
    main_rows = [row for row in rows if row["display_mode"] not in baseline_modes]
    baseline_rows = [row for row in rows if row["display_mode"] in baseline_modes]
    main_rows.sort(key=lambda row: row["test_top1"], reverse=True)
    baseline_order = {"fp_base": 0, "ptq_base": 1}
    baseline_rows.sort(key=lambda row: baseline_order[row["display_mode"]])

    header = format_row("Type", "mode", "test_top1", "test_loss")
    print(header)
    for row in main_rows:
        print(format_row(row["display_type"], row["display_mode"], f"{row['test_top1']:.2f}", f"{row['test_loss']:.4f}"))
    if baseline_rows:
        print("-" * len(header))
    for row in baseline_rows:
        print(format_row(row["display_type"], row["display_mode"], f"{row['test_top1']:.2f}", f"{row['test_loss']:.4f}"))


if __name__ == "__main__":
    main()
