import numpy as np


def call_payoff(ST, K):
    return np.maximum(ST - K, 0.0)


def put_payoff(ST, K):
    return np.maximum(K - ST, 0.0)
