# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# CIFAR-10 channel statistics (computed on the full training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def data_loader(batchsize=128, subset_size=None, num_workers=2, augment=True):
    """
    Load CIFAR-10.

    Args:
        batchsize   : mini-batch size (default 128)
        subset_size : if set, use only the first N training samples
                      (useful for quick smoke tests, e.g. subset_size=2000)
        num_workers : DataLoader workers (set 0 on Windows if multiprocessing crashes)
        augment     : apply RandomCrop + RandomHorizontalFlip on the training set

    Returns:
        train_loader, test_loader, classes
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    if subset_size is not None:
        trainset = Subset(trainset, list(range(subset_size)))

    train_loader = DataLoader(
        trainset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, CLASSES
