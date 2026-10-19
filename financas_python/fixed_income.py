"""Valor do dinheiro no tempo e renda fixa (Capítulos 3 e 8)."""
import numpy as np
from scipy.optimize import brentq


def vp(fluxo, taxa, periodos):
    return fluxo / (1 + taxa) ** periodos


def vpl(fluxos, taxa):
    return float(sum(cf / (1 + taxa) ** t for t, cf in enumerate(fluxos)))


def tir(fluxos):
    try:
        return float(brentq(lambda r: vpl(fluxos, r), -0.999, 100.0, xtol=1e-12))
    except ValueError:
        return float("nan")


def pmt(principal, taxa, n):
    return principal / n if taxa == 0 else principal * taxa / (1 - (1 + taxa) ** (-n))


def tabela_price(principal, taxa, n):
    """Lista de (período, prestação, juros, amortização, saldo)."""
    p, saldo, linhas = pmt(principal, taxa, n), principal, []
    for t in range(1, n + 1):
        juros = saldo * taxa; amort = p - juros; saldo -= amort
        linhas.append((t, p, juros, amort, max(saldo, 0.0)))
    return linhas


def tabela_sac(principal, taxa, n):
    a, saldo, linhas = principal / n, principal, []
    for t in range(1, n + 1):
        juros = saldo * taxa; saldo -= a
        linhas.append((t, a + juros, juros, a, max(saldo, 0.0)))
    return linhas


def preco_titulo(vn, cupom, ytm, anos, freq=2):
    n, r = int(round(anos * freq)), ytm / freq
    c = vn * cupom / freq
    if r == 0:
        return c * n + vn
    return float(c * (1 - (1 + r) ** (-n)) / r + vn / (1 + r) ** n)


def duracao(vn, cupom, ytm, anos, freq=2):
    """(Macaulay, modificada), em anos."""
    n, r = int(round(anos * freq)), ytm / freq
    c = vn * cupom / freq
    t = np.arange(1, n + 1)
    fluxos = np.full(n, c); fluxos[-1] += vn
    P = preco_titulo(vn, cupom, ytm, anos, freq)
    mac = float(np.sum((t / freq) * fluxos / (1 + r) ** t) / P)
    return mac, mac / (1 + r)


def convexidade(vn, cupom, ytm, anos, freq=2, h=1e-4):
    """Convexidade numérica: (P+ + P- - 2P) / (P h^2)."""
    P = preco_titulo(vn, cupom, ytm, anos, freq)
    return float((preco_titulo(vn, cupom, ytm + h, anos, freq)
                  + preco_titulo(vn, cupom, ytm - h, anos, freq) - 2 * P) / (P * h**2))


def ytm(preco, vn, cupom, anos, freq=2):
    return float(brentq(lambda y: preco_titulo(vn, cupom, y, anos, freq) - preco, 1e-6, 1.0, xtol=1e-12))


def taxa_forward(z1, t1, z2, t2):
    return float(((1 + z2) ** t2 / (1 + z1) ** t1) ** (1 / (t2 - t1)) - 1)


def pu_di(taxa_pct, dias_uteis):
    """PU do DI Futuro (100.000 pontos no vencimento); taxa em % a.a."""
    return 100_000 / (1 + taxa_pct / 100) ** (dias_uteis / 252)
