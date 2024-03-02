from abc import ABC, abstractmethod
from typing import Tuple
from torch.utils.data import DataLoader


class DataLoadersGeneratorBase(ABC):
    @abstractmethod
    def get(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError
