import numpy as np
from black_scholes.src.black_scholes import call_price, put_price


def implied_vol_bisect(price_mkt, S, K, T, r, option="call", a=1e-6, b=3.0, tol=1e-8, nmax=200):
    """
    Vol implicite par dichotomie (bisection).
    On cherche sigma tel que BS_price(sigma) = price_mkt.
    """
    price_fn = call_price if option == "call" else put_price

    def f(sigma):
        return price_fn(S, K, T, r, sigma) - price_mkt

    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Intervalle [a,b] incorrect : f(a) et f(b) ont le même signe. Élargis a/b.")

    for _ in range(nmax):
        m = 0.5 * (a + b)
        fm = f(m)

        if abs(fm) < tol or (b - a) < tol:
            return m

        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return 0.5 * (a + b)


def implied_vol_newton(price_mkt, S, K, T, r, option="call", sigma0=0.2, tol=1e-8, nmax=50):
    """
    Vol implicite par Newton.
    Plus rapide, mais peut échouer si la pente (vega) est trop faible.
    """
    price_fn = call_price if option == "call" else put_price

    sigma = float(sigma0)
    eps = 1e-5  # petit bump sur sigma pour approx dérivée

    for _ in range(nmax):
        price = price_fn(S, K, T, r, sigma)
        err = price - price_mkt
        if abs(err) < tol:
            return sigma

        # dérivée numérique dPrice/dSigma (≈ vega)
        price_up = price_fn(S, K, T, r, sigma + eps)
        price_dn = price_fn(S, K, T, r, max(sigma - eps, 1e-12))
        vega = (price_up - price_dn) / (2 * eps)

        if abs(vega) < 1e-10:
            raise RuntimeError("Newton instable : vega trop faible (option très ITM/OTM ou T petite).")

        sigma = sigma - err / vega

    return sigma
