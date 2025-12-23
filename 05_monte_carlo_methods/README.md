# Monte Carlo Methods

This project focuses on **Monte Carlo methods for option pricing**, using the
Black–Scholes framework as a probabilistic model and analytical benchmark.

The objective is to study Monte Carlo as a **numerical pricing tool**: its
convergence properties, statistical error, and practical behavior when
estimating option prices.

## What is covered
- Monte Carlo pricing of European call options under a GBM assumption
- Exact simulation of terminal asset prices
- Construction of confidence intervals using the Central Limit Theorem
- Empirical study of Monte Carlo convergence
- Comparison with the Black–Scholes analytical price

## Approach
The project relies on the exact distribution of the terminal price under
Geometric Brownian Motion, avoiding time discretization.

Monte Carlo estimates are obtained by simulating independent terminal prices,
computing option payoffs, and discounting them under the risk-neutral measure.
Confidence intervals are used to quantify the statistical uncertainty of the
estimator.

Results are compared with the Black–Scholes closed-form solution to illustrate
Monte Carlo convergence and the characteristic 1/sqrt(N) error decay.

## Main takeaways
- Monte Carlo provides unbiased estimators but converges slowly
- Statistical uncertainty must always be quantified using confidence intervals
- Black–Scholes serves as a reliable benchmark to validate numerical methods
- Reproducibility is essential when analyzing Monte Carlo behavior

## Outputs
- Monte Carlo price estimates for European call options
- Confidence intervals at different simulation budgets
- Convergence plots comparing Monte Carlo estimates to Black–Scholes prices

## Limitations
- Simple GBM dynamics with constant parameters
- European payoffs only
- No variance reduction techniques
- Synthetic data only (no market calibration)

## Next steps
- Variance reduction techniques to improve numerical efficiency
- Monte Carlo pricing of path-dependent and exotic options
- Extensions to more advanced stochastic models
"""