import numpy as np


def standard_error(x):
    """
    Erreur standard de la moyenne empirique.
    """
    return np.std(x, ddof=1) / np.sqrt(len(x))


def confidence_interval(price, stderr, alpha=0.05):
    """
    Intervalle de confiance asymptotique (CLT).
    """
    z = 1.96  # 95%
    return price - z * stderr, price + z * stderr
