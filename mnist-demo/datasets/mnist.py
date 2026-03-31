from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_data_loaders(
    data_root: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    # 这个 demo 只研究三种训练方式的差异：
    # 1. 浮点训练
    # 2. 量化训练
    # 3. 量化训练 + QAS
    #
    # 所以数据处理保持最简单，只做 ToTensor()。
    # 不额外加归一化和数据增强，是为了避免实验变量太多，
    # 这样更容易把精度差异归因到模型/优化方式本身。
    transform = transforms.ToTensor()

    # train=True 读取训练集，train=False 读取测试集。
    # train.py 只消费训练集，evaluate.py 只消费测试集。
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

