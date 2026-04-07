import torch
from torch.utils.data import DataLoader, TensorDataset

from mnist_data import load_mnist_float


def build_data_loaders(
    data_root: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    # 数据层改成共享的 NumPy 读取逻辑。
    # 这样 torch_impl 和后续 numpy_impl 会消费同一份原始 IDX 数据，
    # 避免两边的数据预处理路径悄悄分叉。
    train_images, train_labels = load_mnist_float(data_root, train=True)
    test_images, test_labels = load_mnist_float(data_root, train=False)

    train_set = TensorDataset(
        torch.from_numpy(train_images),
        torch.from_numpy(train_labels),
    )
    test_set = TensorDataset(
        torch.from_numpy(test_images),
        torch.from_numpy(test_labels),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
