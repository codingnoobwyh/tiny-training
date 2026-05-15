from pathlib import Path

import numpy as np

from common.dataset import DEFAULT_DATA_ROOT, load_mnist_float


def load_mnist_for_numpy(data_root: str | Path | None = None, train: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT
    else:
        data_root = Path(data_root)

    return load_mnist_float(data_root, train=train, add_channel_dim=True)
