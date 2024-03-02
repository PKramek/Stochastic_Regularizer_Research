from typing import Union

import torch


class MetricBase:
    def calculate(self, X: torch.Tensor, Y: torch.Tensor) -> Union[int, float]:
        raise NotImplementedError
