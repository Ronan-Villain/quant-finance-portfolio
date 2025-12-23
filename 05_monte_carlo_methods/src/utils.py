import numpy as np


def discount(payoffs, r, T):
    return np.exp(-r * T) * payoffs
