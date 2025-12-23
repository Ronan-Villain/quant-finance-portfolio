import numpy as np

from black_scholes.src.black_scholes import call_price
from implied_volatility.src.implied_volatility import implied_vol_bisection


def make_synthetic_market_prices(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    base_sigma: float = 0.2,
    smile_strength: float = 0.3,
) -> np.ndarray:
    """
    Crée des "prix marché" synthétiques avec un smile artificiel :
    sigma(K) = base_sigma * (1 + smile_strength * |ln(K/S)|)

    Ça permet de construire un smile sans dataset réel.
    """
    strikes = np.asarray(strikes, dtype=float)

    sigmas = base_sigma * (1.0 + smile_strength * np.abs(np.log(strikes / S)))
    prices = np.array([call_price(S, K, T, r, sig) for K, sig in zip(strikes, sigmas)])
    return prices


def implied_vol_smile_from_prices(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    market_prices: np.ndarray,
) -> np.ndarray:
    """
    Calcule la vol implicite pour chaque strike à partir des prix marché.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_prices = np.asarray(market_prices, dtype=float)

    iv = []
    for K, p in zip(strikes, market_prices):
        iv.append(
            implied_vol_bisection(
                market_price=float(p),
                S=S,
                K=float(K),
                T=T,
                r=r,
                option_type="call",
                vol_low=1e-6,
                vol_high=3.0,
            )
        )
    return np.array(iv, dtype=float)
