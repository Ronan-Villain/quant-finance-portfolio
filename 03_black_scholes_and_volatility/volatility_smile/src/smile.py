import numpy as np

from black_scholes.src.black_scholes import call_price
from implied_volatility.src.implied_volatility import implied_vol_bisect


def synthetic_call_prices(S, strikes, T, r, sigma0=0.2, smile=0.3):
    """
    Génère des prix de call avec une volatilité dépendant du strike.
    """
    strikes = np.asarray(strikes)

    sigmas = sigma0 * (1 + smile * np.abs(np.log(strikes / S)))
    prices = [call_price(S, K, T, r, sig) for K, sig in zip(strikes, sigmas)]

    return np.array(prices)


def implied_vol_smile(S, strikes, T, r, prices):
    """
    Calcule la volatilité implicite pour chaque strike.
    """
    strikes = np.asarray(strikes)

    iv = [
        implied_vol_bisect(p, S, K, T, r, option="call")
        for K, p in zip(strikes, prices)
    ]

    return np.array(iv)
