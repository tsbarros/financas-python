"""Teoria de carteiras (Capítulos 2, 4 e 5)."""
import numpy as np
from scipy.optimize import minimize


def retorno_carteira(w, mu):
    return float(np.asarray(w) @ np.asarray(mu))


def variancia_carteira(w, Sigma):
    w = np.asarray(w)
    return float(w @ np.asarray(Sigma) @ w)


def _restricoes(n, sem_venda_a_descoberto):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n if sem_venda_a_descoberto else [(None, None)] * n
    return cons, bounds


def minima_variancia(Sigma, sem_venda_a_descoberto=True):
    """Carteira de mínima variância global (GMV). Devolve o vetor de pesos."""
    Sigma = np.asarray(Sigma)
    n = len(Sigma)
    cons, bounds = _restricoes(n, sem_venda_a_descoberto)
    res = minimize(variancia_carteira, np.ones(n) / n, args=(Sigma,), method="SLSQP",
                   bounds=bounds, constraints=cons, options={"ftol": 1e-12})
    return res.x


def maximo_sharpe(mu, Sigma, rf=0.0, sem_venda_a_descoberto=True):
    """Carteira tangente (máximo índice de Sharpe)."""
    mu, Sigma = np.asarray(mu), np.asarray(Sigma)
    n = len(mu)
    cons, bounds = _restricoes(n, sem_venda_a_descoberto)

    def neg_sharpe(w):
        return -(retorno_carteira(w, mu) - rf) / np.sqrt(variancia_carteira(w, Sigma))

    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"ftol": 1e-12})
    return res.x


def fronteira_eficiente(mu, Sigma, n_pontos=50, sem_venda_a_descoberto=True):
    """Pontos da fronteira eficiente: devolve (retornos_alvo, volatilidades, pesos)."""
    mu, Sigma = np.asarray(mu), np.asarray(Sigma)
    n = len(mu)
    w_gmv = minima_variancia(Sigma, sem_venda_a_descoberto)
    alvos = np.linspace(retorno_carteira(w_gmv, mu), mu.max(), n_pontos)
    cons0, bounds = _restricoes(n, sem_venda_a_descoberto)
    rets, vols, pesos = [], [], []
    for alvo in alvos:
        cons = cons0 + [{"type": "eq", "fun": lambda w, a=alvo: retorno_carteira(w, mu) - a}]
        res = minimize(variancia_carteira, np.ones(n) / n, args=(Sigma,), method="SLSQP",
                       bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
        if res.success:
            rets.append(alvo); vols.append(np.sqrt(res.fun)); pesos.append(res.x)
    return np.array(rets), np.array(vols), np.array(pesos)


def paridade_de_risco(Sigma):
    """Carteira de paridade de risco (contribuições iguais ao risco total)."""
    Sigma = np.asarray(Sigma)
    n = len(Sigma)

    def obj(w):
        rc = w * (Sigma @ w)
        return np.sum((rc - rc.mean()) ** 2)

    cons, bounds = _restricoes(n, True)
    res = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-16, "maxiter": 1000})
    return res.x / res.x.sum()


def beta_capm(ret_ativo, ret_mercado, rf_periodo=0.0):
    """Estima (alpha, beta) do CAPM por MQO sobre excessos de retorno."""
    y = np.asarray(ret_ativo) - rf_periodo
    x = np.asarray(ret_mercado) - rf_periodo
    X = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(alpha), float(beta)
