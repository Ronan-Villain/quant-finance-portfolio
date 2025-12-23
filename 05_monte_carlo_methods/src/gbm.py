import numpy as np


def simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed=None):
    """
    Simulation exacte de S_T sous GBM.

    S_T = S0 * exp((r - 0.5*sigma^2)T + sigma*sqrt(T)*Z)
    """
    seed = np.random.randint(0, 1_000_000)

    Z = np.random.randn(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return ST
