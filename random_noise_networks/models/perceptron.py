import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerPerceptron(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super(ThreeLayerPerceptron, self).__init__()
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
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size)

    def forward(self, x):
        x = self._fc1(x)
        x = F.relu(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._fc3(x)
        x = F.relu(x)
        output = F.softmax(x, dim=1)
        return output
