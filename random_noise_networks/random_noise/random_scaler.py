from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class NormalRandomScaler(nn.Module):
    def __init__(self, p: float, std: float):
        super(NormalRandomScaler, self).__init__()
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

    def _get_normaly_distributed_scales(self, size: Tuple[int, ...]) -> Tensor:
        # Generate samples from a normal distribution with mean equal to 1.0 and std equal to <self._std>
        random_samples = torch.empty(size).normal_(mean=1.0, std=self._std)

        return random_samples

    def forward(self, input: Tensor) -> Tensor:
        noisy_input = input
        if self.training:  # Apply noise only during training
            # Use pytorch random number generator, so that you only have to set one seed manually
            random_number = torch.rand(1).item()

            if random_number <= self._p:
                with torch.no_grad():
                    random_scale = self._get_normaly_distributed_scales(
                        size=input.size()
                    )
                    random_scale = random_scale.to(input.device)
                    noisy_input = input.mul(random_scale)

        return noisy_input


class MaskedNormalRandomScaler(NormalRandomScaler):

    def _get_normaly_distributed_scales(self, size: Tuple[int, ...]) -> Tensor:
        # Generate samples from a normal distribution with mean equal to 1.0 and std equal to <self._std>
        random_samples = torch.empty(size).normal_(mean=1.0, std=self._std)

        return random_samples

    def _get_random_binary_mask(self, size: Tuple[int, ...]) -> Tensor:
        # Generates random binary mask, where the probability that the value will be equal to 1 is <self._p>
        random_binary_mask = torch.rand(size) <= self._p

        return random_binary_mask

    def forward(self, input: Tensor) -> Tensor:
        noisy_input = input
        if self.training:  # Apply noise only during training
            with torch.no_grad():
                random_scaler = self._get_normaly_distributed_scales(size=input.size())
                scaler_mask = self._get_random_binary_mask(size=input.size())
                negated_scaler_mask = torch.logical_not(scaler_mask)
                # When we are not applying a random scaler, we multiply by 1. Thus we must add negated mask
                multiplier = (random_scaler * scaler_mask) + negated_scaler_mask

                multiplier = multiplier.to(input.device)
                noisy_input = input.mul(multiplier)

        return noisy_input


class UniformRandomScaler(nn.Module):
    def __init__(self, p: float, range_start: float, range_end: float):
        super(UniformRandomScaler, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"Random scaling probability has to be between 0 and 1, but got {p}"
            )
        if range_start < 0:
            raise ValueError(
                f"Range start must be grater than 0, but got {range_start}"
            )
        if range_start > range_end:
            raise ValueError(
                f"Range start ({range_start}) must be smaller than the range end ({range_end})"
            )

        self._p = p
        self._range_start = range_start
        self._range_end = range_end

    def _get_uniformly_distributed_scales(self, size: Tuple[int, ...]) -> Tensor:
        random_samples = torch.empty(size).uniform_(self._range_start, self._range_end)

        return random_samples

    def forward(self, input: Tensor) -> Tensor:
        noisy_input = input
        if self.training:  # Apply noise only during training
            # Use pytorch random number generator, so that you only have to set one seed manually
            random_number = torch.rand(1).item()

            if random_number <= self._p:
                with torch.no_grad():
                    random_scale = self._get_uniformly_distributed_scales(
                        size=input.size()
                    )
                    random_scale = random_scale.to(input.device)
                    noisy_input = input.mul(random_scale)

        return noisy_input
