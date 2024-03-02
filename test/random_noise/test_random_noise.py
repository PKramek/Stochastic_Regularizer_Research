import pytest
import torch
from random_noise_networks.random_noise import NormalRandomNoise
from scipy import stats


@pytest.mark.parametrize(
    "input_size, std",
    [
        (8192, 1e-4),
        (8192, 1e-3),
    ],
)
def test_random_noise_is_normal(input_size, std):
    input = torch.zeros(input_size)
    random_noise = NormalRandomNoise(p=1.0, std=std)

    output = random_noise(input)
    sample_std, sample_mean = torch.std_mean(output)

    _, p_value = stats.kstest(output, "norm")

    assert p_value <= 0.01
    assert sample_std == pytest.approx(std, abs=1e-4)
    assert sample_mean == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize(
    "input_size",
    [
        16,
        64,
    ],
)
def test_random_noise_does_not_modify_input_during_evaluation(input_size):
    input = torch.zeros(input_size)
    expected_output = torch.clone(input)
    random_noise = NormalRandomNoise(p=1.0, std=1e-3)

    random_noise.eval()
    output = random_noise(input)

    assert torch.equal(output, expected_output)
