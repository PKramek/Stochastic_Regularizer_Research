from random_noise_networks.experiment_objective.experiment_objective_base import (
    ExperimentObjectiveBase,
)

import optuna
import torch.nn as nn


from random_noise_networks.models import (
    DropoutThreeLayerPerceptron,
    RandomNoiseThreeLayerPerceptron,
    RandomScalingThreeLayerPerceptron,
    RandomMaskedScalingThreeLayerPerceptron,
    RandomUniformScalingThreeLayerPerceptron,
    RandomScalingWithDropoutThreeLayerPerceptron,
    ThreeLayerPerceptron,
)


class ThreeLayerPerceptronExperimentObjective(ExperimentObjectiveBase):
    def get_model(self, trial: optuna.trial.Trial) -> nn.Module:
        HIDDEN_SIZE = 1024
        model = ThreeLayerPerceptron(
            hidden_size=HIDDEN_SIZE,
            input_size=self._input_size,
            output_size=self._output_size,
        )

        return model


class DropoutThreeLayerPerceptronExperimentObjective(ExperimentObjectiveBase):
    def get_model(self, trial: optuna.trial.Trial) -> nn.Module:
        dropout_prob = trial.suggest_float("dropout_prob", low=0.1, high=1.0, step=0.1)

        HIDDEN_SIZE = 1024
        model = DropoutThreeLayerPerceptron(
            dropout_prob=dropout_prob,
            hidden_size=HIDDEN_SIZE,
            input_size=self._input_size,
            output_size=self._output_size,
        )

        return model


class RandomNoiseThreeLayerPerceptronExperimentObjective(ExperimentObjectiveBase):
    def get_model(self, trial: optuna.trial.Trial) -> nn.Module:
        random_noise_prob = trial.suggest_float(
            "random_noise_prob", low=0.1, high=1.0, step=0.1
        )
        random_noise_std = trial.suggest_float("std", low=1e-7, high=1e-3)

        HIDDEN_SIZE = 1024
        model = RandomNoiseThreeLayerPerceptron(
            hidden_size=HIDDEN_SIZE,
            input_size=self._input_size,
            output_size=self._output_size,
            random_noise_prob=random_noise_prob,
            std=random_noise_std,
        )

        return model


class RandomScalingThreeLayerPerceptronExperimentObjective(ExperimentObjectiveBase):
    def get_model(self, trial: optuna.trial.Trial) -> nn.Module:
        random_scaling_prob = trial.suggest_float(
            "random_scaling_prob", low=0.1, high=1.0, step=0.1
        )
        random_scaling_std = trial.suggest_float("std", low=1e-5, high=1.0)

        HIDDEN_SIZE = 1024
        model = RandomScalingThreeLayerPerceptron(
            hidden_size=HIDDEN_SIZE,
            input_size=self._input_size,
            output_size=self._output_size,
            random_scaling_prob=random_scaling_prob,
            std=random_scaling_std,
        )

        return model


class RandomMaskedScalingThreeLayerPerceptronExperimentObjective(
    ExperimentObjectiveBase
):
    def get_model(self, trial: optuna.trial.Trial) -> nn.Module:
        random_scaling_prob = trial.suggest_float(
            "random_scaling_prob", low=0.1, high=1.0, step=0.1
        )
        random_scaling_std = trial.suggest_float("std", low=1e-5, high=1.0)

        HIDDEN_SIZE = 1024
        model = RandomMaskedScalingThreeLayerPerceptron(
            hidden_size=HIDDEN_SIZE,
            input_size=self._input_size,
            output_size=self._output_size,
            random_scaling_prob=random_scaling_prob,
            std=random_scaling_std,
        )

        return model


class RandomUniformScalingThreeLayerPerceptronExperimentObjective(
    ExperimentObjectiveBase
):
    def get_model(self, trial: optuna.trial.Trial) -> nn.Module:
        random_scaling_prob = trial.suggest_float(
            "random_scaling_prob", low=0.1, high=0.8, step=0.1
        )
        random_scaling_range_end = trial.suggest_float(
            "random_scaling_range_end", low=0, high=1.0, step=0.1
        )

        HIDDEN_SIZE = 1024
        model = RandomUniformScalingThreeLayerPerceptron(
            hidden_size=HIDDEN_SIZE,
            input_size=self._input_size,
            output_size=self._output_size,
            random_scaling_prob=random_scaling_prob,
            range_end=random_scaling_range_end,
            range_start=0.0,
        )

        return model


class RandomScalingWithDropoutThreeLayerPerceptronExperimentObjective(
    ExperimentObjectiveBase
):
    def get_model(self, trial: optuna.trial.Trial) -> nn.Module:
        random_scaling_prob = trial.suggest_float(
            "random_scaling_prob", low=0.1, high=1.0, step=0.1
        )
        random_scaling_std = trial.suggest_float("std", low=1e-5, high=2.0, step=0.1)
        dropout_prob = trial.suggest_float("dropout_prob", low=0.1, high=1.0, step=0.1)

        HIDDEN_SIZE = 1024
        model = RandomScalingWithDropoutThreeLayerPerceptron(
            hidden_size=HIDDEN_SIZE,
            input_size=self._input_size,
            output_size=self._output_size,
            dropout_prob=dropout_prob,
            std=random_scaling_std,
            random_scaling_prob=random_scaling_prob,
        )

        return model
