from typing import Union

import torch


class AccuracyMetric:
    def calculate(self, Y_PRED: torch.Tensor, Y: torch.Tensor) -> Union[int, float]:
        correct = torch.sum(Y_PRED == Y).item()
        acc = correct / torch.numel(Y)

        return acc
