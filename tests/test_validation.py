import numpy as np
import pytest
import scipy.sparse
import torch

from binaria.validation import to_output, validate_binary_matrix


@pytest.mark.parametrize(
    "value",
    [
        [[0, 1], [1, 0]],
        np.array([[0, 1], [1, 0]]),
        torch.tensor([[0, 1], [1, 0]]),
        scipy.sparse.csr_matrix([[0, 1], [1, 0]]),
    ],
)
def test_binary_inputs_are_coerced_to_the_requested_dtype(value: object) -> None:
    result = validate_binary_matrix(value, dtype=torch.float32)
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([[0.0, np.nan]], "NaN"),
        ([[0.0, np.inf]], "infinity"),
        ([[0.0, -1.0]], "Negative values"),
        ([[0.0, 0.5]], "outside"),
    ],
)
def test_invalid_values_have_specific_errors(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_binary_matrix(value)


@pytest.mark.parametrize("value", [[0, 1], np.empty((0, 2)), np.empty((2, 0))])
def test_invalid_shapes_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        validate_binary_matrix(value)


def test_to_output_detaches_without_moving_torch_results() -> None:
    value = torch.ones(2, requires_grad=True)
    tensor = to_output(value, output="torch")
    array = to_output(value, output="numpy")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device == value.device
    assert tensor.requires_grad is False
    assert isinstance(array, np.ndarray)


def test_to_output_rejects_unknown_formats() -> None:
    with pytest.raises(ValueError, match="Unknown output format"):
        to_output(torch.ones(1), output="list")  # type: ignore[arg-type]
