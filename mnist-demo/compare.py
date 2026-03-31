import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    # 把多个 run 的最终评估结果并排打印
    parser = argparse.ArgumentParser(description="Compare evaluated MNIST demo runs")
    parser.add_argument("--output-root", default="mnist-demo/artifacts/runs")
    parser.add_argument("--run-names", nargs="+", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = []

    for run_name in args.run_names:
        eval_path = output_root / run_name / "eval_result.json"
        result = load_json(eval_path)
        rows.append(result)

    # 按测试精度从高到低排序，方便一眼看到当前最好的方法。
    rows.sort(key=lambda row: row["test_top1"], reverse=True)

    print("run_name\tmode\ttest_top1\ttest_loss")
    for row in rows:
        print(
            f"{row['run_name']}\t{row['mode']}\t"
            f"{row['test_top1']:.2f}\t{row['test_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
