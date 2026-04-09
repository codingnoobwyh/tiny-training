"""
MNIST 数据加载模块 - 最小核心
"""

from pathlib import Path

import numpy as np

# 导入共享的数据加载模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from mnist_data import load_mnist_float


def load_mnist_for_numpy(data_root: str | Path | None = None, train: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    加载 MNIST 数据，返回 NumPy 格式

    Args:
        data_root: 数据根目录，默认为 PROJECT_ROOT/data
        train: 是否加载训练集(False 为测试集)

    Returns:
        (images, labels) 元组
        - images: shape (N, 1, 28, 28), dtype=float32, range=[0.0, 1.0]
        - labels: shape (N,), dtype=int64, range=[0, 9]
    """
    if data_root is None:
        data_root = PROJECT_ROOT / "data"
    else:
        data_root = Path(data_root)

    return load_mnist_float(data_root, train=train, add_channel_dim=True)
