# %%
import logging
from typing import Dict, Type

import optuna
import torch
import torch.nn.functional as F
from random_noise_networks.data_loaders_generator.mnist_data_loaders_generator import (
    MNISTDataLoadersGenerator,
)

from random_noise_networks.experiment_objective import (
    ExperimentObjectiveBase,
    ThreeLayerPerceptronExperimentObjective,
    DropoutThreeLayerPerceptronExperimentObjective,
    RandomScalingThreeLayerPerceptronExperimentObjective,
    RandomUniformScalingThreeLayerPerceptronExperimentObjective,
    RandomNoiseThreeLayerPerceptronExperimentObjective,
    RandomScalingWithDropoutThreeLayerPerceptronExperimentObjective,
)

logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

# %%
EXPERIMENT_THREE_LAYER_PERCEPTRON = "Three Layer Perceptron"
EXPERIMENT_DROPOUT_THREE_LAYER_PERCEPTRON = "Dropout Three Layer Perceptron"
EXPERIMENT_RANDOM_NOISE_THREE_LAYER_PERCEPTRON = "Random Noise Three Layer Perceptron"
EXPERIMENT_RANDOM_SCALING_THREE_LAYER_PERCEPTRON = (
    "Random Scaling Three Layer Perceptron"
)
EXPERIMENT_RANDOM_SCALING_WITH_DROPOUT_THREE_LAYER_PERCEPTRON = (
    "Random Scaling with Dropout Three Layer Perceptron"
)

EXPERIMENT_RANDOM_UNIFORM_SCALING_THREE_LAYER_PERCEPTRON = (
    "Random Uniform Scaling Three Layer Perceptron"
)


EXPERIMENTS: Dict[str, Type[ExperimentObjectiveBase]] = {
    EXPERIMENT_THREE_LAYER_PERCEPTRON: ThreeLayerPerceptronExperimentObjective,
    EXPERIMENT_DROPOUT_THREE_LAYER_PERCEPTRON: DropoutThreeLayerPerceptronExperimentObjective,
    EXPERIMENT_RANDOM_NOISE_THREE_LAYER_PERCEPTRON: RandomNoiseThreeLayerPerceptronExperimentObjective,
    EXPERIMENT_RANDOM_SCALING_THREE_LAYER_PERCEPTRON: RandomScalingThreeLayerPerceptronExperimentObjective,
    EXPERIMENT_RANDOM_UNIFORM_SCALING_THREE_LAYER_PERCEPTRON: RandomUniformScalingThreeLayerPerceptronExperimentObjective,
    EXPERIMENT_RANDOM_SCALING_WITH_DROPOUT_THREE_LAYER_PERCEPTRON: RandomScalingWithDropoutThreeLayerPerceptronExperimentObjective,
}


HIDDE_SIZE = 1024
NUMBER_OF_EPOCHS = 20
INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10

N_TRIALS = 500 * 2
TIMEOUT = 5 * 60 * 60

USE_CUDA = True
if USE_CUDA and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

logger.info(f"Using device: {DEVICE}")
# %%

if __name__ == "__main__":
    CURRENT_EXPERIMENT_NAME = (
        EXPERIMENT_RANDOM_SCALING_WITH_DROPOUT_THREE_LAYER_PERCEPTRON
    )
    CURRENT_EXPERIMENT_OBJECTIVE = EXPERIMENTS[CURRENT_EXPERIMENT_NAME]

    logger.info(f"Running experiment: {CURRENT_EXPERIMENT_NAME}")

    experiment_objective = EXPERIMENTS[CURRENT_EXPERIMENT_NAME](
        criterion=F.cross_entropy,
        device=DEVICE,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        logger=logger,
        number_of_epochs=NUMBER_OF_EPOCHS,
        train_validation_dataloader_generator=MNISTDataLoadersGenerator(),
    )

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name=CURRENT_EXPERIMENT_NAME,
        load_if_exists=True,
    )
    study.optimize(experiment_objective.run, n_trials=N_TRIALS, timeout=TIMEOUT)
    logger.info(f"Best value: {study.best_value} (params: {study.best_params})")

# %%
