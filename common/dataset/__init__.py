from pathlib import Path

from common.dataset.loaders import build_data_loaders
from common.dataset.mnist import export_mnist_fp32_payload, load_mnist_float

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
