import numpy as np

# On réutilise ton BS comme "pricing engine"

from black_scholes.src.black_scholes import call_price, put_price


def implied_vol_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    vol_low: float = 1e-6,
    vol_high: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Implied volatility via bisection.
    Robust and simple (slower than Newton, but very stable).

    Parameters
    ----------
    market_price : observed option price
    option_type  : "call" or "put"
    vol_low/high : bracket for sigma
    """
    if market_price <= 0:
        raise ValueError("market_price must be > 0")

    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    price_fn = call_price if option_type == "call" else put_price

    # Objective: f(sigma) = model_price(sigma) - market_price
    def f(sig):
        return price_fn(S, K, T, r, sig) - market_price

    a, b = float(vol_low), float(vol_high)
    fa, fb = f(a), f(b)

    # Ensure we have a bracket: fa and fb must have opposite signs
    if fa * fb > 0:
        raise ValueError(
            "Bisection bracket failed: try widening [vol_low, vol_high]. "
            f"f(vol_low)={fa:.6g}, f(vol_high)={fb:.6g}"
        )

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)

        if abs(fm) < tol or (b - a) < tol:
            return m

        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return 0.5 * (a + b)


def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    sigma0: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """
    Implied volatility via Newton-Raphson.
    Faster, but may fail if initial guess is poor or vega is tiny.

    Uses numerical derivative w.r.t sigma (finite differences) to avoid
    dependency on greeks implementation.
    """
    if market_price <= 0:
        raise ValueError("market_price must be > 0")

    option_type = option_type.lower().strip()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    price_fn = call_price if option_type == "call" else put_price

    sigma = float(sigma0)
    eps = 1e-5  # finite-diff step for derivative in sigma

    for _ in range(max_iter):
        price = price_fn(S, K, T, r, sigma)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # numerical vega: dPrice/dSigma
        price_up = price_fn(S, K, T, r, sigma + eps)
        price_dn = price_fn(S, K, T, r, max(sigma - eps, 1e-12))
        vega_num = (price_up - price_dn) / (2 * eps)

        if abs(vega_num) < 1e-10:
            # vega too small -> Newton unstable
            raise RuntimeError("Newton failed: numerical vega too small.")

        sigma = sigma - diff / vega_num

        # Keep sigma in a reasonable range
        sigma = max(1e-12, min(5.0, sigma))

    return sigma
