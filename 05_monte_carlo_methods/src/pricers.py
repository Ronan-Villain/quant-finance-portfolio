import numpy as np

from src.gbm import simulate_gbm_terminal
from src.payoffs import call_payoff, put_payoff
from src.stats import standard_error, confidence_interval
from src.utils import discount


def mc_call_price_gbm(S0, K, r, sigma, T, n_paths, seed=None):
    """
    Pricing Monte Carlo d'un call européen sous GBM.
    Retourne (prix, intervalle de confiance 95%).
    """
    ST = simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed)
    payoffs = call_payoff(ST, K)
    discounted = discount(payoffs, r, T)

    price = np.mean(discounted)
    stderr = standard_error(discounted)
    ci = confidence_interval(price, stderr)

    return price, ci


def mc_put_price_gbm(S0, K, r, sigma, T, n_paths, seed=None):
    """
    Pricing Monte Carlo d'un put européen sous GBM.
    """
    ST = simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed)
    payoffs = put_payoff(ST, K)
    discounted = discount(payoffs, r, T)

    price = np.mean(discounted)
    stderr = standard_error(discounted)
    ci = confidence_interval(price, stderr)

    return price, ci
