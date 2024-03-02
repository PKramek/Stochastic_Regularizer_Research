import random
from typing import Tuple

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

from random_noise_networks.data_loaders_generator.data_loaders_generator_base import (
    DataLoadersGeneratorBase,
)


class MNISTDataLoadersGenerator(DataLoadersGeneratorBase):
    def get(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Normalize arguments explanation: https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
                transforms.Normalize((0.1307,), (0.3081,)),
                torch.flatten,
            ]
        )
        mnist_trainset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )
        # Manually set seed to 42 so that the train and test split is always deterministic
        INITIAL_SEED = torch.initial_seed()
        torch.manual_seed(42)

        TRAIN_SIZE, VALIDATION_SIZE = 50000, 10000
        train_set, val_set = torch.utils.data.random_split(  # type: ignore
            mnist_trainset,
            [TRAIN_SIZE, VALIDATION_SIZE],
        )
        torch.manual_seed(INITIAL_SEED)

        NUM_WORKERS = 16
        train_data_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
        )
        validation_data_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
        )

        return train_data_loader, validation_data_loader
