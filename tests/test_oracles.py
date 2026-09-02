import pytest
import torch

from binaria._core import Core
from tests.oracles.naive import log_likelihood as naive_log_likelihood


@pytest.mark.parametrize(
    ("n_samples", "n_features", "n_components"),
    [
        (4, 3, 2),
        (10, 4, 2),  # samples >> latent factors -- the realistic shape
        (2, 2, 2),  # square, easy to get "accidentally right"
    ],
)
def test_log_likelihood_matches_naive(n_samples: int, n_features: int, n_components: int) -> None:
    # This is what pins the pi = sigmoid(-beta @ energy) sign convention: a
    # flipped sign in forward() would disagree with naive.py, which encodes
    # Eq 1 independently and has no shared code with Core.
    torch.manual_seed(0)
    core = Core(n_samples=n_samples, n_features=n_features, n_components=n_components)
    data = torch.randint(0, 2, (n_samples, n_features)).double()

    vectorized = core.log_likelihood(data).item()
    naive = naive_log_likelihood(core.beta.tolist(), core.energy.tolist(), data.tolist())

    assert vectorized == pytest.approx(naive, abs=1e-10)


def test_masked_log_likelihood_matches_naive() -> None:
    torch.manual_seed(0)
    core = Core(n_samples=4, n_features=3, n_components=2)
    data = torch.randint(0, 2, (4, 3)).double()
    mask = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=torch.float64)

    vectorized = core.log_likelihood(data, mask=mask).item()
    naive = naive_log_likelihood(
        core.beta.tolist(), core.energy.tolist(), data.tolist(), mask.tolist()
    )

    assert vectorized == pytest.approx(naive, abs=1e-10)
