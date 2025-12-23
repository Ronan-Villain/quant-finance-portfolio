import numpy as np
from scipy.stats import norm

def d1(S, K, T, r, sigma):
    """
    d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    """
    d2 = d1 - sigma*sqrt(T)
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    """
    European call option price under Black–Scholes (no dividends).
    """
    dd1 = d1(S, K, T, r, sigma)
    dd2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(dd1) - K * np.exp(-r * T) * norm.cdf(dd2)


def put_price(S, K, T, r, sigma):
    """
    European put option price under Black–Scholes (no dividends).
    """
    dd1 = d1(S, K, T, r, sigma)
    dd2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-dd2) - S * norm.cdf(-dd1)


def discount_factor(T, r):
    """
    Discount factor: exp(-rT)
    """
    return np.exp(-r * T)


def forward_price(S, T, r):
    """
    Forward price (no dividends): F = S * exp(rT)
    """
    return S * np.exp(r * T)
