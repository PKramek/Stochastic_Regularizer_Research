from logging import Logger
from typing import Callable, Dict, Optional
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import torch.optim as optim

from tqdm import tqdm

from random_noise_networks.data_loaders_generator.data_loaders_generator_base import (
    DataLoadersGeneratorBase,
)


class ExperimentObjectiveBase(ABC):
    BATCH_SIZE_CHOICES = [8, 16, 32, 64, 128]
    LEARNING_RATE_LOW = 1e-5
    LEARNING_RATE_HIGH = 1e-3

    def __init__(
        self,
        train_validation_dataloader_generator: DataLoadersGeneratorBase,
        number_of_epochs: int,
        criterion: Callable,
        input_size: int,
        output_size: int,
        device: torch.device,
        logger: Logger,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        self._train_validation_dataloader_generator = (
            train_validation_dataloader_generator
        )
        self._criterion = criterion
        self._input_size = input_size
        self._output_size = output_size
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._logger = logger
        self._device = device
        self._nuber_of_epochs = number_of_epochs

    @abstractmethod
    def get_model(self, trial: optuna.trial.Trial) -> nn.Module:
        raise NotImplementedError

    def run(self, trial: optuna.trial.Trial) -> float:
        learning_rate = self._learning_rate or trial.suggest_float(
            "learning_rate",
            low=self.LEARNING_RATE_LOW,
            high=self.LEARNING_RATE_HIGH,
            log=True,
        )
        batch_size = self._batch_size or trial.suggest_categorical(
            "batch_size", choices=self.BATCH_SIZE_CHOICES
        )

        model = self.get_model(trial)
        model.to(device=self._device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        (
            train_loader,
            validation_loader,
        ) = self._train_validation_dataloader_generator.get(batch_size=batch_size)

        max_accuracy = 0.0
        for epoch in range(1, self._nuber_of_epochs + 1):
            self.train(model=model, optimizer=optimizer, train_loader=train_loader)
            test_results = self.evaluate(
                model=model, validation_loader=validation_loader
            )

            self._logger.info(test_results)
            if test_results["Accuracy"] > max_accuracy:
                max_accuracy = test_results["Accuracy"]

            trial.report(test_results["Accuracy"], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return max_accuracy

    def train(self, model: nn.Module, optimizer, train_loader: DataLoader) -> None:
        model.train()
        with tqdm(train_loader) as t:
            for data, target in t:
                data, target = data.to(self._device), target.to(self._device)
                optimizer.zero_grad()
                output = model(data)
                loss = self._criterion(output, target, reduction="mean")
                loss.backward()
                optimizer.step()

    def evaluate(
        self, model: nn.Module, validation_loader: DataLoader
    ) -> Dict[str, float]:
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(self._device), target.to(self._device)
                output = model(data)
                test_loss += self._criterion(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        # TODO move to using multiple metrics
        test_loss /= len(validation_loader.dataset)  # type: ignore
        accuracy = 100.0 * correct / len(validation_loader.dataset)  # type: ignore

        return {
            "Average loss": test_loss,
            "Accuracy": accuracy,
        }

    def __call__(self, trial: optuna.trial.Trial) -> float:
        return self.run(trial=trial)
