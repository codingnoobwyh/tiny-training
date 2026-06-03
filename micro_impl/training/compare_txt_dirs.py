#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CompareResult:
    name: str
    count: int | str
    same: bool
    cosine: float | str
    l2: float | str
    max_abs: float | str
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare same-name .txt numeric dumps in two directories."
    )
    parser.add_argument("left", help="first directory")
    parser.add_argument("right", help="second directory")
    return parser.parse_args()


def collect_txt_files(root: Path) -> dict[str, Path]:
    return {path.name: path for path in root.glob("*.txt") if path.is_file()}


def load_numeric_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64, ndmin=1)
    return np.asarray(data, dtype=np.float64).reshape(-1)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def compare_file(name: str, left_path: Path, right_path: Path, rtol: float, atol: float) -> CompareResult:
    try:
        left = load_numeric_txt(left_path)
        right = load_numeric_txt(right_path)
    except Exception as exc:
        return CompareResult(name, "-", False, "-", "-", "-", f"load failed: {exc}")

    if left.shape != right.shape:
        return CompareResult(
            name,
            f"{left.size}/{right.size}",
            False,
            "-",
            "-",
            "-",
            f"shape differs: {left.shape} vs {right.shape}",
        )

    diff = left - right
    same = bool(np.allclose(left, right, rtol=rtol, atol=atol))
    return CompareResult(
        name=name,
        count=left.size,
        same=same,
        cosine=cosine_similarity(left, right),
        l2=float(np.linalg.norm(diff)),
        max_abs=float(np.max(np.abs(diff))) if diff.size else 0.0,
    )


def format_float(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.9g}"


def print_results(results: list[CompareResult], missing_left: list[str], missing_right: list[str]) -> None:
    print(f"{'file':<42} {'count':>10} {'same':>6} {'cosine':>14} {'l2':>14} {'max_abs':>14}  reason")
    print("-" * 124)
    for result in results:
        print(
            f"{result.name:<42} {str(result.count):>10} {str(result.same):>6} "
            f"{format_float(result.cosine):>14} {format_float(result.l2):>14} "
            f"{format_float(result.max_abs):>14}  {result.reason}"
        )

    print("-" * 124)
    print(f"common: {len(results)}, missing_in_left: {len(missing_left)}, missing_in_right: {len(missing_right)}")
    if missing_left:
        print("missing in left:")
        for name in missing_left:
            print(f"  {name}")
    if missing_right:
        print("missing in right:")
        for name in missing_right:
            print(f"  {name}")


def main() -> None:
    args = parse_args()
    left_root = Path(args.left).resolve()
    right_root = Path(args.right).resolve()
    if not left_root.is_dir():
        raise NotADirectoryError(left_root)
    if not right_root.is_dir():
        raise NotADirectoryError(right_root)

    left_files = collect_txt_files(left_root)
    right_files = collect_txt_files(right_root)
    common_names = sorted(set(left_files) & set(right_files))
    missing_left = sorted(set(right_files) - set(left_files))
    missing_right = sorted(set(left_files) - set(right_files))

    results = [
        compare_file(name, left_files[name], right_files[name], rtol=1e-5, atol=1e-8)
        for name in common_names
    ]
    print_results(results, missing_left, missing_right)


if __name__ == "__main__":
    main()
