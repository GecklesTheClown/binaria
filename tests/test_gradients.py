import pytest
import torch
import torch.nn.functional as F

from binaria._core import Core


@pytest.mark.parametrize(
    ("n_samples", "n_features", "n_components"),
    [
        (6, 5, 3),
        (20, 4, 2),  # samples >> latent factors -- the realistic shape
        (2, 2, 2),  # square, easy to get "accidentally right"
    ],
)
def test_analytic_gradient_matches_autograd(
    n_samples: int, n_features: int, n_components: int
) -> None:
    torch.manual_seed(0)
    core = Core(n_samples=n_samples, n_features=n_features, n_components=n_components)
    data = torch.randint(0, 2, (n_samples, n_features)).double()

    core.zero_grad()
    core.log_likelihood(data).backward()
    assert core.beta.grad is not None
    assert core.energy.grad is not None
    autograd_beta = core.beta.grad.clone()
    autograd_energy = core.energy.grad.clone()

    analytic_beta, analytic_energy = core.analytic_gradients(data)

    assert torch.allclose(autograd_beta, analytic_beta, atol=1e-10)
    assert torch.allclose(autograd_energy, analytic_energy, atol=1e-10)


@pytest.mark.parametrize(
    ("n_samples", "n_features", "n_components"),
    [
        (4, 3, 2),
        (10, 4, 2),  # samples >> latent factors -- the realistic shape
        (2, 2, 2),
    ],
)
def test_autograd_gradient_matches_finite_differences(
    n_samples: int, n_features: int, n_components: int
) -> None:
    # Catches: errors in the loss definition itself. This is a genuinely
    # different check from test_analytic_gradient_matches_autograd above --
    # that test compares two things that both differentiate the SAME
    # forward pass (Eq 3's hand-derived formula vs autograd), so a bug in
    # forward() itself would make both of them agree while both are wrong.
    # gradcheck's finite-difference approximation doesn't depend on either
    # differentiation path -- it only depends on forward() actually
    # computing the right thing -- so it's a genuinely independent oracle,
    # the same role naive.py plays for the log-likelihood itself.
    #
    # torch.autograd.gradcheck needs autograd.grad(output, inputs) to reach
    # the exact tensor objects it perturbs. Core(beta=..., energy=...)
    # can't be used directly for this: nn.Parameter(beta) creates a new
    # leaf tensor that breaks the graph connection back to the external
    # beta (confirmed empirically -- torch.autograd.grad raises "not used
    # in the graph" through a plain Core construction). torch.func.
    # functional_call is the standard fix: it calls Core.forward()'s real
    # code with externally-supplied parameters while preserving the graph,
    # so this tests the actual forward() implementation, not a
    # reimplementation of it.
    torch.manual_seed(0)
    core = Core(n_samples=n_samples, n_features=n_features, n_components=n_components)
    beta = torch.randn(n_samples, n_components, dtype=torch.float64, requires_grad=True)
    energy = torch.randn(n_components, n_features, dtype=torch.float64, requires_grad=True)
    data = torch.randint(0, 2, (n_samples, n_features)).double()

    def log_likelihood_via_core(beta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        logits = torch.func.functional_call(core, {"beta": beta, "energy": energy}, args=())
        return -F.binary_cross_entropy_with_logits(logits, data, reduction="sum")

    assert torch.autograd.gradcheck(log_likelihood_via_core, (beta, energy))
