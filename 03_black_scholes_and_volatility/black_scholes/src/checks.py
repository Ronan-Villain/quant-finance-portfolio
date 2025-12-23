import numpy as np
from .black_scholes import call_price, put_price
from .greeks import delta_call


def put_call_parity_error(S, K, T, r, sigma):
    """
    Put-call parity (no dividends):
    C - P = S - K * exp(-rT)

    Returns the parity error (should be close to 0).
    """
    C = call_price(S, K, T, r, sigma)
    P = put_price(S, K, T, r, sigma)
    rhs = S - K * np.exp(-r * T)
    return (C - P) - rhs


def numerical_delta_call(S, K, T, r, sigma, h=1e-4):
    """
    Central finite difference approximation of call delta.
    """
    return (call_price(S + h, K, T, r, sigma) - call_price(S - h, K, T, r, sigma)) / (2.0 * h)


def check_delta_consistency(S, K, T, r, sigma, h=1e-4):
    """
    Compare analytical delta vs numerical delta.
    Returns (analytical, numerical, difference).
    """
    da = delta_call(S, K, T, r, sigma)
    dn = numerical_delta_call(S, K, T, r, sigma, h=h)
    return da, dn, (da - dn)
