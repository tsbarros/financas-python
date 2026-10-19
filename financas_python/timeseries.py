"""Séries temporais financeiras (Capítulos 1, 2, 10 e 18)."""
import numpy as np
from scipy import stats


def retornos_log(precos):
    p = np.asarray(precos, dtype=float)
    return np.diff(np.log(p))


def retornos_simples(precos):
    p = np.asarray(precos, dtype=float)
    return p[1:] / p[:-1] - 1


def estatisticas(ret, periodos_ano=252):
    r = np.asarray(ret)
    return {"media_anual": r.mean() * periodos_ano,
            "vol_anual": r.std(ddof=1) * np.sqrt(periodos_ano),
            "assimetria": float(stats.skew(r)),
            "curtose_excesso": float(stats.kurtosis(r)),
            "minimo": r.min(), "maximo": r.max()}


def sharpe(ret, rf_periodo=0.0, periodos_ano=252):
    ex = np.asarray(ret) - rf_periodo
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(periodos_ano))


def drawdown_maximo(ret_log):
    cum = np.exp(np.cumsum(ret_log))
    return float((cum / np.maximum.accumulate(cum) - 1).min())


def autocorrelacao(x, lags):
    x = np.asarray(x) - np.mean(x)
    den = np.sum(x**2)
    return np.array([np.sum(x[k:] * x[:-k]) / den for k in range(1, lags + 1)])


def ljung_box(x, lags=10):
    """Estatística Q e p-valor (qui-quadrado com `lags` g.l.)."""
    n = len(x)
    rho = autocorrelacao(x, lags)
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, lags + 1)))
    return float(q), float(1 - stats.chi2.cdf(q, df=lags))


def variancia_garch(omega, alpha, beta, residuos, var0=None):
    """Variância condicional GARCH(1,1) para uma série de resíduos."""
    e = np.asarray(residuos)
    s2 = np.empty_like(e, dtype=float)
    s2[0] = var0 if var0 is not None else e.var()
    for t in range(1, len(e)):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
    return s2


def previsao_variancia_garch(omega, alpha, beta, var_t1, h):
    """E_t[sigma^2_{t+h}] converge para omega/(1-alpha-beta)."""
    longo = omega / (1 - alpha - beta)
    return longo + (alpha + beta) ** (h - 1) * (var_t1 - longo)
