"""Medidas de risco (Capítulo 7). Convenção: retornos em decimal; VaR e CVaR como perdas positivas."""
import numpy as np
from scipy import stats


def var_historico(retornos, alpha=0.99, valor=1.0):
    r = np.asarray(retornos)
    return float(-np.percentile(r, (1 - alpha) * 100) * valor)


def cvar_historico(retornos, alpha=0.99, valor=1.0):
    """Expected Shortfall histórico: média das perdas na cauda além do VaR."""
    r = np.asarray(retornos)
    corte = np.percentile(r, (1 - alpha) * 100)
    cauda = r[r <= corte]
    return float(-cauda.mean() * valor)


def var_parametrico(retornos, alpha=0.99, valor=1.0):
    r = np.asarray(retornos)
    z = stats.norm.ppf(1 - alpha)
    return float(-(r.mean() + z * r.std(ddof=1)) * valor)


def cvar_parametrico(retornos, alpha=0.99, valor=1.0):
    r = np.asarray(retornos)
    z = stats.norm.ppf(alpha)
    return float((-r.mean() + r.std(ddof=1) * stats.norm.pdf(z) / (1 - alpha)) * valor)


def kupiec_pof(n_violacoes, n_dias, alpha=0.99):
    """Teste de Kupiec (POF). Devolve (LR, p-valor)."""
    p = 1 - alpha
    if n_violacoes in (0, n_dias):
        return 0.0, 1.0
    pi = n_violacoes / n_dias
    ll0 = (n_dias - n_violacoes) * np.log(1 - p) + n_violacoes * np.log(p)
    ll1 = (n_dias - n_violacoes) * np.log(1 - pi) + n_violacoes * np.log(pi)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def christoffersen_independencia(violacoes):
    """Teste de independência de Christoffersen. Devolve (LR, p-valor)."""
    v = np.asarray(violacoes)
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0)); n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0)); n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(p01, p11):
        s = 0.0
        if n00 and p01 < 1: s += n00 * np.log(1 - p01)
        if n01 and p01 > 0: s += n01 * np.log(p01)
        if n10 and p11 < 1: s += n10 * np.log(1 - p11)
        if n11 and p11 > 0: s += n11 * np.log(p11)
        return s

    lr = -2 * (ll(pi, pi) - ll(pi01, pi11))
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))
