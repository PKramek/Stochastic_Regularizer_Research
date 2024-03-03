import torch.nn as nn
import torch.nn.functional as F

from random_noise_networks.random_noise import NormalRandomNoise
from random_noise_networks.random_noise.random_scaler import (
    NormalRandomScaler,
    UniformRandomScaler,
    MaskedNormalRandomScaler,
)


class RandomNoiseThreeLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        random_noise_prob: float,
        std: float,
    ):
        super(RandomNoiseThreeLayerPerceptron, self).__init__()
        if hidden_size < 0:
            raise ValueError(
                f"Hidden size must be grater than 0, but got {hidden_size}"
            )
        if input_size < 0:
            raise ValueError(f"Input size must be grater than 0, but got {input_size}")
        if output_size < 0:
            raise ValueError(
                f"Output size must be grater than 0, but got {output_size}"
            )
        if random_noise_prob < 0 or random_noise_prob > 1:
            raise ValueError(
                f"Random noise application probability has to be between 0 and 1, but got {random_noise_prob}"
            )
        if std < 0:
            raise ValueError(
                f"Random noise std has to be greater than 0, but got {random_noise_prob}"
            )
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._random_noise_prob = random_noise_prob
        self._std = std

        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size)

        self._random_noise1 = NormalRandomNoise(
            p=self._random_noise_prob, std=self._std
        )
        self._random_noise2 = NormalRandomNoise(
            p=self._random_noise_prob, std=self._std
        )

    def forward(self, x):
        x = self._fc1(x)
        x = F.relu(x)
        x = self._random_noise1(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._random_noise2(x)
        x = self._fc3(x)
        output = F.softmax(x, dim=1)

        return output


class RandomMaskedScalingThreeLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        random_scaling_prob: float,
        std: float,
    ):
        super(RandomMaskedScalingThreeLayerPerceptron, self).__init__()
        if hidden_size < 0:
            raise ValueError(
                f"Hidden size must be grater than 0, but got {hidden_size}"
            )
        if input_size < 0:
            raise ValueError(f"Input size must be grater than 0, but got {input_size}")
        if output_size < 0:
            raise ValueError(
                f"Output size must be grater than 0, but got {output_size}"
            )
        if random_scaling_prob < 0 or random_scaling_prob > 1:
            raise ValueError(
                f"Random scaling application probability has to be between 0 and 1, but got {random_scaling_prob}"
            )
        if std < 0:
            raise ValueError(
                f"Random noise std has to be greater than 0, but got {random_scaling_prob}"
            )
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._random_scaling_prob = random_scaling_prob
        self._std = std

        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size)

        self._random_scaler1 = MaskedNormalRandomScaler(
            p=self._random_scaling_prob, std=self._std
        )
        self._random_scaler2 = MaskedNormalRandomScaler(
            p=self._random_scaling_prob, std=self._std
        )

    def forward(self, x):
        x = self._fc1(x)
        x = F.relu(x)
        x = self._random_scaler1(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._random_scaler2(x)
        x = self._fc3(x)
        output = F.softmax(x, dim=1)

        return output


class RandomMaskedFirstLayerPreScalingThreeLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        random_scaling_prob: float,
        std: float,
    ):
        super(RandomMaskedFirstLayerPreScalingThreeLayerPerceptron, self).__init__()
        if hidden_size < 0:
            raise ValueError(
                f"Hidden size must be grater than 0, but got {hidden_size}"
            )
        if input_size < 0:
            raise ValueError(f"Input size must be grater than 0, but got {input_size}")
        if output_size < 0:
            raise ValueError(
                f"Output size must be grater than 0, but got {output_size}"
            )
        if random_scaling_prob < 0 or random_scaling_prob > 1:
            raise ValueError(
                f"Random scaling application probability has to be between 0 and 1, but got {random_scaling_prob}"
            )
        if std < 0:
            raise ValueError(
                f"Random noise std has to be greater than 0, but got {random_scaling_prob}"
            )
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._random_scaling_prob = random_scaling_prob
        self._std = std

        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size)

        self._random_scaler1 = MaskedNormalRandomScaler(
            p=self._random_scaling_prob, std=self._std
        )

    def forward(self, x):
        x = self._fc1(x)
        x = self._random_scaler1(x)
        x = F.relu(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._fc3(x)
        output = F.softmax(x, dim=1)

        return output


class RandomScalingThreeLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        random_scaling_prob: float,
        std: float,
    ):
        super(RandomScalingThreeLayerPerceptron, self).__init__()
        if hidden_size < 0:
            raise ValueError(
                f"Hidden size must be grater than 0, but got {hidden_size}"
            )
        if input_size < 0:
            raise ValueError(f"Input size must be grater than 0, but got {input_size}")
        if output_size < 0:
            raise ValueError(
                f"Output size must be grater than 0, but got {output_size}"
            )
        if random_scaling_prob < 0 or random_scaling_prob > 1:
            raise ValueError(
                f"Random scaling application probability has to be between 0 and 1, but got {random_scaling_prob}"
            )
        if std < 0:
            raise ValueError(
                f"Random noise std has to be greater than 0, but got {random_scaling_prob}"
            )
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._random_scaling_prob = random_scaling_prob
        self._std = std

        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size)

        self._random_scaler1 = NormalRandomScaler(
            p=self._random_scaling_prob, std=self._std
        )
        self._random_scaler2 = NormalRandomScaler(
            p=self._random_scaling_prob, std=self._std
        )

    def forward(self, x):
        x = self._fc1(x)
        x = F.relu(x)
        x = self._random_scaler1(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._random_scaler2(x)
        x = self._fc3(x)
        output = F.softmax(x, dim=1)

        return output


class RandomUniformScalingThreeLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        random_scaling_prob: float,
        range_start: float,
        range_end: float,
    ):
        super(RandomUniformScalingThreeLayerPerceptron, self).__init__()
        if hidden_size < 0:
            raise ValueError(
                f"Hidden size must be grater than 0, but got {hidden_size}"
            )
        if input_size < 0:
            raise ValueError(f"Input size must be grater than 0, but got {input_size}")
        if output_size < 0:
            raise ValueError(
                f"Output size must be grater than 0, but got {output_size}"
            )
        if random_scaling_prob < 0 or random_scaling_prob > 1:
            raise ValueError(
                f"Random scaling application probability has to be between 0 and 1, but got {random_scaling_prob}"
            )
        if range_start < 0:
            raise ValueError(
                f"Range start must be grater than 0, but got {range_start}"
            )
        if range_start > range_end:
            raise ValueError(
                f"Range start ({range_start}) must be smaller than the range end ({range_end})"
            )

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._random_scaling_prob = random_scaling_prob
        self._range_start = range_start
        self._range_end = range_end

        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size)

        self._random_scaler1 = UniformRandomScaler(
            p=self._random_scaling_prob,
            range_start=self._range_start,
            range_end=self._range_end,
        )
        self._random_scaler2 = UniformRandomScaler(
            p=self._random_scaling_prob,
            range_start=self._range_start,
            range_end=self._range_end,
        )

    def forward(self, x):
        x = self._fc1(x)
        x = F.relu(x)
        x = self._random_scaler1(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._random_scaler2(x)
        x = self._fc3(x)
        output = F.softmax(x, dim=1)

        return output


class RandomScalingWithDropoutThreeLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout_prob: float,
        random_scaling_prob: float,
        std: float,
    ):
        super(RandomScalingWithDropoutThreeLayerPerceptron, self).__init__()
        if hidden_size < 0:
            raise ValueError(
                f"Hidden size must be grater than 0, but got {hidden_size}"
            )
        if input_size < 0:
            raise ValueError(f"Input size must be grater than 0, but got {input_size}")
        if output_size < 0:
            raise ValueError(
                f"Output size must be grater than 0, but got {output_size}"
            )
        if random_scaling_prob < 0 or random_scaling_prob > 1:
            raise ValueError(
                f"Random scaling application probability has to be between 0 and 1, but got {random_scaling_prob}"
            )
        if dropout_prob < 0 or dropout_prob > 1:
            raise ValueError(
                f"Dropout probability has to be between 0 and 1, but got {dropout_prob}"
            )
        if std < 0:
            raise ValueError(
                f"Random noise std has to be greater than 0, but got {random_scaling_prob}"
            )
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._dropout_prob = dropout_prob
        self._random_scaling_prob = random_scaling_prob
        self._std = std

        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size)

        self._random_scaler1 = NormalRandomScaler(
            p=self._random_scaling_prob, std=self._std
        )
        self._random_scaler2 = NormalRandomScaler(
            p=self._random_scaling_prob, std=self._std
        )

        self._dropout1 = nn.Dropout(p=self._dropout_prob)
        self._dropout2 = nn.Dropout(p=self._dropout_prob)

    def forward(self, x):
        x = self._fc1(x)
        x = F.relu(x)
        x = self._random_scaler1(x)
        x = self._dropout1(x)

        x = self._fc2(x)
        x = F.relu(x)
        x = self._random_scaler2(x)
        x = self._dropout2(x)

        x = self._fc3(x)
        output = F.softmax(x, dim=1)

        return output
