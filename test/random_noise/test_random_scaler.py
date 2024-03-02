import pytest
import torch
from random_noise_networks.random_noise import NormalRandomScaler
from scipy import stats


@pytest.mark.parametrize(
    "input_size, std",
    [
        (1024, 1e-2),
        (8192, 1e-3),
        (8192, 1e-4),
    ],
)
def test_random_scaler_scales_using_normal_distribution(input_size, std):
    input = torch.ones(input_size)
    random_scaler = NormalRandomScaler(p=1.0, std=std)

    output = random_scaler(input)
    output_scales = output.div(input)

    sample_std, sample_mean = torch.std_mean(output_scales)

    _, p_value = stats.kstest(output, "norm")

    assert p_value <= 0.01
    assert sample_std == pytest.approx(std, abs=1e-3)
    assert sample_mean == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    "input_size",
    [
        16,
        64,
    ],
)
def test_random_scaler_does_not_modify_input_during_evaluation(input_size):
    input = torch.ones(input_size)
    expected_output = torch.clone(input)
    random_noise = NormalRandomScaler(p=1.0, std=1)

    random_noise.eval()
    output = random_noise(input)

    assert torch.equal(output, expected_output)
