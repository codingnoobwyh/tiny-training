import struct
from pathlib import Path

import numpy as np


TRAIN_IMAGES = "train-images-idx3-ubyte"
TRAIN_LABELS = "train-labels-idx1-ubyte"
TEST_IMAGES = "t10k-images-idx3-ubyte"
TEST_LABELS = "t10k-labels-idx1-ubyte"


def _raw_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "MNIST" / "raw"


def _read_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST image magic number in {path}: {magic}")
        images = np.frombuffer(handle.read(), dtype=np.uint8)
    return images.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected MNIST label magic number in {path}: {magic}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    return labels.reshape(count)


def ensure_mnist_downloaded(data_root: str | Path) -> None:
    raw_dir = _raw_dir(data_root)
    expected = [
        raw_dir / TRAIN_IMAGES,
        raw_dir / TRAIN_LABELS,
        raw_dir / TEST_IMAGES,
        raw_dir / TEST_LABELS,
    ]
    if all(path.exists() for path in expected):
        return

    # 这里用 torchvision 只负责下载原始 IDX 文件。
    # 真正的数据读取走本文件里的 NumPy 逻辑，保证 torch_impl 和 numpy_impl
    # 后续都能共享同一套原始数据入口。
    from torchvision import datasets

    datasets.MNIST(root=str(data_root), train=True, download=True)
    datasets.MNIST(root=str(data_root), train=False, download=True)


def load_mnist_uint8(data_root: str | Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    ensure_mnist_downloaded(data_root)
    raw_dir = _raw_dir(data_root)
    image_name = TRAIN_IMAGES if train else TEST_IMAGES
    label_name = TRAIN_LABELS if train else TEST_LABELS
    images = _read_idx_images(raw_dir / image_name)
    labels = _read_idx_labels(raw_dir / label_name)
    return images, labels


def load_mnist_float(
    data_root: str | Path,
    train: bool,
    add_channel_dim: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    images, labels = load_mnist_uint8(data_root, train=train)
    images = images.astype(np.float32) / 255.0
    if add_channel_dim:
        images = images[:, None, :, :]
    labels = labels.astype(np.int64)
    return images, labels
