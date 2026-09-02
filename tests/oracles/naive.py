import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def probability_matrix(beta: list[list[float]], energy: list[list[float]]) -> list[list[float]]:
    k = len(energy)
    n_features = len(energy[0])
    n_samples = len(beta)

    matrix = [[0.0] * n_features for _ in range(n_samples)]

    for s in range(n_samples):
        for i in range(n_features):
            dot_product = sum(
                beta[s][latent_dim] * energy[latent_dim][i] for latent_dim in range(k)
            )
            matrix[s][i] = sigmoid(-dot_product)
    return matrix


def log_likelihood(
    beta: list[list[float]],
    energy: list[list[float]],
    observations: list[list[float]],
    mask: list[list[float]] | None = None,
) -> float:
    prob_matrix = probability_matrix(beta, energy)
    log_likelihood_value = 0.0

    for s in range(len(observations)):
        for i in range(len(observations[0])):
            if mask is not None and mask[s][i] == 0:
                continue
            if observations[s][i] == 1:
                log_likelihood_value += math.log(prob_matrix[s][i])
            else:
                log_likelihood_value += math.log(1 - prob_matrix[s][i])

    return log_likelihood_value
