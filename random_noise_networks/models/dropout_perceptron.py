import torch.nn as nn
import torch.nn.functional as F


class DropoutThreeLayerPerceptron(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int, dropout_prob: float
    ):
        super(DropoutThreeLayerPerceptron, self).__init__()
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
        if dropout_prob < 0 or dropout_prob > 1:
            raise ValueError(
                f"Dropout probability has to be between 0 and 1, but got {dropout_prob}"
            )
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._dropout_prob = dropout_prob

        self._fc1 = nn.Linear(self._input_size, self._hidden_size)
        self._fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self._fc3 = nn.Linear(self._hidden_size, self._output_size)

        self._dropout1 = nn.Dropout(p=self._dropout_prob)
        self._dropout2 = nn.Dropout(p=self._dropout_prob)

    def forward(self, x):
        x = self._fc1(x)
        x = F.relu(x)
        x = self._dropout1(x)
        x = self._fc2(x)
        x = F.relu(x)
        x = self._dropout2(x)
        x = self._fc3(x)
        output = F.softmax(x, dim=1)
        return output
