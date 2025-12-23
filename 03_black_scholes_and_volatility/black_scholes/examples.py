from src.black_scholes import call_price, put_price
from src.checks import put_call_parity_error, check_delta_consistency
from src.greeks import delta_call, gamma, vega

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

C = call_price(S, K, T, r, sigma)
P = put_price(S, K, T, r, sigma)

print("Call:", C)
print("Put :", P)
print("Put-call parity error:", put_call_parity_error(S, K, T, r, sigma))

print("Delta(call):", delta_call(S, K, T, r, sigma))
print("Gamma      :", gamma(S, K, T, r, sigma))
print("Vega       :", vega(S, K, T, r, sigma))

da, dn, diff = check_delta_consistency(S, K, T, r, sigma, h=1e-4)
print("Delta analytical:", da)
print("Delta numerical :", dn)
print("Delta diff      :", diff)
