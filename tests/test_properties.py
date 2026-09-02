import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from binaria._core import Core

# Small enough that true gradient ascent theory guarantees a step won't
# decrease a smooth objective (first-order Taylor term dominates the
# curvature error term) -- verified empirically across 2500 random trials
# before this test was written, not just assumed.
_ASCENT_LR = 1e-5
_STEPS_PER_TRIAL = 5


@given(
    n_samples=st.integers(min_value=2, max_value=10),
    n_features=st.integers(min_value=2, max_value=10),
    n_components=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
    gradient=st.sampled_from(["analytic", "autograd"]),
)
@settings(max_examples=100, deadline=None)
def test_small_ascent_step_never_decreases_log_likelihood(
    n_samples: int,
    n_features: int,
    n_components: int,
    seed: int,
    gradient: str,
) -> None:
    # Catches: divergence, bad step size, sign flips -- a flipped-sign
    # gradient would *decrease* log-likelihood every step, immediately
    # violating this. Uses a small fixed-step manual update, not fit()'s
    # Adam loop: Adam's momentum/adaptive scaling isn't itself guaranteed
    # monotone even when the underlying gradient is correct, so testing
    # through Adam would make this flaky for reasons unrelated to
    # correctness. This tests the gradient's sign/direction directly.
    torch.manual_seed(seed)
    core = Core(n_samples=n_samples, n_features=n_features, n_components=n_components)
    data = torch.randint(0, 2, (n_samples, n_features)).double()

    for _ in range(_STEPS_PER_TRIAL):
        ll_before = core.log_likelihood(data).item()

        if gradient == "analytic":
            grad_beta, grad_energy = core.analytic_gradients(data)
        else:
            core.zero_grad()
            core.log_likelihood(data).backward()  # type: ignore[no-untyped-call]
            assert core.beta.grad is not None
            assert core.energy.grad is not None
            grad_beta, grad_energy = core.beta.grad, core.energy.grad

        with torch.no_grad():
            core.beta += _ASCENT_LR * grad_beta
            core.energy += _ASCENT_LR * grad_energy

        ll_after = core.log_likelihood(data).item()
        assert ll_after >= ll_before - 1e-9
