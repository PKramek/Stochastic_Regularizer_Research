from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class NormalRandomNoise(nn.Module):
    def __init__(self, p: float, std: float):
        super(NormalRandomNoise, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"Normal random noise probability has to be between 0 and 1, but got {p}"
            )
        if std < 0:
            raise ValueError(
                f"Normal random noise standard deviation has to be greater than 0, but got {std}"
            )

        self._p = p
        self._std = std

    def _get_normal_noise(self, size: Tuple[int, ...]) -> Tensor:
        # Generate samples from a normal distribution with mean equal to 0.0 and std equal to <self._std>
        random_samples = torch.empty(size).normal_(mean=0.0, std=self._std)

        return random_samples

    def forward(self, input: Tensor) -> Tensor:
        noisy_input = input
        if self.training:  # Apply noise only during training
            # Use pytorch random number generator, so that you only have to set one seed manually
            random_number = torch.rand(1).item()

            if random_number <= self._p:
                with torch.no_grad():
                    random_noise = self._get_normal_noise(size=input.size())
                    random_noise = random_noise.to(input.device)
                    noisy_input = input + random_noise

        return noisy_input
