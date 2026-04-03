from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_data_loaders(
    data_root: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    # 数据层保持最小，只做 ToTensor()。
    # 这里不引入数据增强和额外归一化，避免把主比较从
    # “训练路线差异”混成“数据 recipe 差异”。
    transform = transforms.ToTensor()

    # train=True 读取训练集，train=False 读取测试集。
    # 各 runner 会按自己的职责分别消费 train/test loader。
    train_set = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
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
