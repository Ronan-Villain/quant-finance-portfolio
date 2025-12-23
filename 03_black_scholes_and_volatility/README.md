# Black–Scholes and Volatility

This project explores option pricing starting from the Black–Scholes model
and moves toward market concepts such as implied volatility and volatility
smiles.

Black–Scholes is used here as a reference model to understand the link
between option prices and volatility, rather than as a realistic market
model.

## What is covered
- Analytical pricing of European call and put options
- Numerical computation of implied volatility from option prices
- Construction of volatility smiles using synthetic market data

## Approach
The project starts with closed-form Black–Scholes pricing formulas.
Option prices are then inverted numerically to recover implied volatility,
which allows the study of volatility smiles and their dependence on strike.

Synthetic prices are used to illustrate how volatility smiles naturally
emerge when the constant-volatility assumption is relaxed.

## Main takeaways
- Volatility is not directly observable and must be inferred from option prices
- Black–Scholes provides a useful analytical benchmark
- Volatility smiles highlight the limitations of constant-volatility models

## Outputs
- Black–Scholes prices for European options
- Implied volatility curves (smiles)
- Figures illustrating the limitations of the model

## Limitations
- Constant volatility assumption
- No path-dependent or exotic payoffs
- Synthetic data only (no real market calibration)

## Next steps
- Sensitivity analysis with respect to model parameters
- Monte Carlo pricing and variance reduction techniques
- Pricing of path-dependent and exotic options
