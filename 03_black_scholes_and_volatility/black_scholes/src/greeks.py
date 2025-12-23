import numpy as np
from scipy.stats import norm
from .black_scholes import d1, d2


def delta_call(S, K, T, r, sigma):
    """
    Delta(call) = N(d1)
    """
    return norm.cdf(d1(S, K, T, r, sigma))


def delta_put(S, K, T, r, sigma):
    """
    Delta(put) = N(d1) - 1
    """
    return norm.cdf(d1(S, K, T, r, sigma)) - 1.0


def gamma(S, K, T, r, sigma):
    """
    Gamma = N'(d1) / (S*sigma*sqrt(T))
    Same for call and put.
    """
    dd1 = d1(S, K, T, r, sigma)
    return norm.pdf(dd1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Vega = dV/dsigma = S*N'(d1)*sqrt(T)
    Same for call and put.
    """
    dd1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(dd1) * np.sqrt(T)


def theta_call(S, K, T, r, sigma):
    """
    Theta(call) (per year).
    """
    dd1 = d1(S, K, T, r, sigma)
    dd2 = d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(dd1) * sigma) / (2.0 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(dd2)
    return term1 + term2


def theta_put(S, K, T, r, sigma):
    """
    Theta(put) (per year).
    """
    dd1 = d1(S, K, T, r, sigma)
    dd2 = d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(dd1) * sigma) / (2.0 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-dd2)
    return term1 + term2


def rho_call(S, K, T, r, sigma):
    """
    Rho(call) = dC/dr
    """
    dd2 = d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(dd2)


def rho_put(S, K, T, r, sigma):
    """
    Rho(put) = dP/dr
    """
    dd2 = d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-dd2)
