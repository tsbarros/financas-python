"""Opções: binomial CRR, Black-Scholes-Merton, gregas e volatilidade implícita (Capítulo 16)."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes(S, K, T, r, sigma, q=0.0, tipo="call"):
    if T <= 0:
        return max(S - K, 0.0) if tipo == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def gregas(S, K, T, r, sigma, q=0.0, tipo="call"):
    """Delta, gamma, vega (por 1% de vol.), theta (por dia corrido) e rho (por 1%)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    phi = norm.pdf(d1)
    gamma = np.exp(-q * T) * phi / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * phi * np.sqrt(T) / 100
    base = -(S * sigma * np.exp(-q * T) * phi) / (2 * np.sqrt(T))
    if tipo == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (base - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (base + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho)}


def binomial(S, K, T, r, sigma, n=500, tipo="call", americana=False):
    """Árvore binomial CRR (sem dividendos). Aceita exercício antecipado."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    desc = np.exp(-r * dt)
    j = np.arange(n + 1)
    precos = S * u ** (n - j) * d ** j
    v = np.maximum(precos - K, 0) if tipo == "call" else np.maximum(K - precos, 0)
    for i in range(n - 1, -1, -1):
        v = desc * (q * v[:-1] + (1 - q) * v[1:])
        if americana:
            j = np.arange(i + 1)
            p = S * u ** (i - j) * d ** j
            ex = np.maximum(p - K, 0) if tipo == "call" else np.maximum(K - p, 0)
            v = np.maximum(v, ex)
    return float(v[0])


def vol_implicita(preco, S, K, T, r, q=0.0, tipo="call"):
    """Inverte Black-Scholes. Devolve NaN se o preço violar os limites de arbitragem."""
    piso = (max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if tipo == "call"
            else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0))
    if preco <= piso + 1e-10:
        return float("nan")
    try:
        return float(brentq(lambda s: black_scholes(S, K, T, r, s, q, tipo) - preco,
                            1e-4, 10.0, xtol=1e-12))
    except ValueError:
        return float("nan")
