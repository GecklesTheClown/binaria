"""
Sign convention: pi = sigmoid(-beta @ energy) (Eq 1). forward() returns the
logits passed to binary_cross_entropy_with_logits, which are -(beta @
energy) -- i.e. -theta, where theta := beta @ energy.

Eq 3's gradients (dL/dbeta, dL/dE) are with respect to theta, not with
respect to the logits (-theta) that binary_cross_entropy_with_logits
actually differentiates through -- these differ in sign. autograd handles
this correctly on its own, since .backward() flows through forward()'s
negation automatically. Hand-written/analytic code (analytic_gradients
below) must apply the same sign flip explicitly; mixing the two
conventions produces a sign error that presents as divergence, not a
crash, which is why this is worth stating up front rather than trusting
each call site to get it right.
"""

import torch
from torch import nn
from torch.nn import functional as F


def normal_init(
    shape: tuple[int, int],
    dtype: torch.dtype,
    std: float = 0.01,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return std * torch.randn(shape, dtype=dtype, device=device, generator=generator)


class Core(nn.Module):
    def __init__(
        self,
        n_samples: int,
        n_features: int,
        n_components: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        beta: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            if beta is not None:
                device = beta.device
            elif energy is not None:
                device = energy.device
        if beta is None:
            beta = normal_init((n_samples, n_components), dtype, device=device, generator=generator)
        if energy is None:
            energy = normal_init(
                (n_components, n_features), dtype, device=device, generator=generator
            )
        self.beta = nn.Parameter(beta)
        self.energy = nn.Parameter(energy)

    def forward(self) -> torch.Tensor:
        # Eq 1: pi = sigmoid(-beta @ energy). The minus sign is the model,
        # not a transcription error -- logits are -(beta @ energy).
        return -(self.beta @ self.energy)

    def log_likelihood(
        self, data: torch.Tensor, *, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self()
        if mask is None:
            return -F.binary_cross_entropy_with_logits(logits, data, reduction="sum")
        per_entry = F.binary_cross_entropy_with_logits(logits, data, reduction="none")
        return -(mask * per_entry).sum()

    @torch.no_grad()
    def analytic_gradients(
        self, data: torch.Tensor, *, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq 3: dL/dbeta_sk = sum_i (pi_si - D_si) E_ki, dL/dE_ki = sum_s
        # (pi_si - D_si) beta_sk -- i.e. dL/dbeta = diff @ energy.T,
        # dL/dE = beta.T @ diff, where diff = pi - D. This is +dL/dparam
        # (ascent direction), matching log_likelihood()'s sign, not the
        # negative-log-likelihood gradient binary_cross_entropy_with_logits
        # would give you. no_grad: this path exists specifically to avoid
        # building an autograd graph every iteration.
        #
        # Masking is the same Eq 3 formula restricted to visible entries,
        # not a new derivation: zeroing the held-out residuals in `diff`
        # before the matmuls drops their contribution to both sums.
        pi = torch.sigmoid(self())
        diff = pi - data
        if mask is not None:
            diff = mask * diff
        grad_beta = diff @ self.energy.T
        grad_energy = self.beta.T @ diff
        return grad_beta, grad_energy

    def l2_penalty(self) -> torch.Tensor:
        """
        ``||beta||_F^2 + ||energy||_F^2`` -- the *unscaled* penalty term.

        The regularization coefficient (``alpha``) lives in ``_optim.fit``,
        not here: this returns the raw quantity so the objective reads
        ``log_likelihood - alpha * l2_penalty``.

        Two algebraic properties are useful for this bilinear model:

        1. ``min`` over factorizations of ``beta @ energy = M`` of this
           quantity equals ``2 * ||M||_nuclear``. So penalizing it is
           exactly nuclear-norm regularization of the logit matrix -- the
           standard convex surrogate for rank, not an ad-hoc shrinkage.
        2. It partially fixes the ``GL(K)`` gauge freedom. The orbit
           ``beta -> beta @ G``, ``energy -> inv(G) @ energy`` leaves
           predictions identical but *changes* this penalty, so minimizing
           it selects the balanced representative where
           ``beta.T @ beta == energy @ energy.T == diag(singular values)``.
           That removes the k(k+1)/2 scaling directions of the gauge,
           leaving only the k(k-1)/2 rotational ones.

        Deliberately not part of ``log_likelihood`` or
        ``analytic_gradients``: those two are Eq 2 and Eq 3, pinned against
        the naive oracle and autograd, and must stay auditable against the
        paper. Regularization is an optimization concern, not a change to
        the model's equations.
        """
        return self.beta.pow(2).sum() + self.energy.pow(2).sum()
