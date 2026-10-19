"""Suíte de testes de financas_python. Execute: python -m pytest -q"""
import numpy as np
import pytest
from financas_python import portfolio, risk, options, fixed_income, timeseries


@pytest.fixture
def mercado():
    mu = np.array([0.12, 0.10, 0.08])
    sig = np.array([0.35, 0.30, 0.25])
    rho = np.array([[1, .45, .30], [.45, 1, .35], [.30, .35, 1]])
    return mu, np.diag(sig) @ rho @ np.diag(sig)


# ---- opções (Capítulo 16)
def test_black_scholes_converge_para_binomial():
    args = dict(S=38.5, K=40.0, T=30 / 252, r=0.105, sigma=0.42)
    assert options.binomial(**args, n=2000) == pytest.approx(options.black_scholes(**args), abs=2e-3)


def test_paridade_put_call():
    S, K, T, r, s, q = 100, 95, 0.5, 0.08, 0.3, 0.02
    c = options.black_scholes(S, K, T, r, s, q, "call"); p = options.black_scholes(S, K, T, r, s, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_call_americana_sem_dividendos_igual_europeia():
    a = options.binomial(100, 100, 1, 0.05, 0.2, 400, "call", americana=True)
    e = options.binomial(100, 100, 1, 0.05, 0.2, 400, "call")
    assert a == pytest.approx(e, abs=1e-8)


def test_put_americana_vale_mais_que_europeia():
    assert options.binomial(100, 110, 1, 0.08, 0.2, 400, "put", americana=True) > \
           options.binomial(100, 110, 1, 0.08, 0.2, 400, "put")


def test_delta_da_call_entre_0_e_1_e_gamma_positivo():
    g = options.gregas(100, 100, 0.5, 0.05, 0.25)
    assert 0 < g["delta"] < 1 and g["gamma"] > 0 and g["vega"] > 0


def test_vol_implicita_recupera_sigma():
    p = options.black_scholes(100, 105, 0.5, 0.06, 0.37)
    assert options.vol_implicita(p, 100, 105, 0.5, 0.06) == pytest.approx(0.37, abs=1e-8)


# ---- risco (Capítulo 7)
def test_cvar_maior_ou_igual_ao_var():
    r = np.random.default_rng(1).standard_t(4, 5000) * 0.01
    for a in (0.95, 0.975, 0.99):
        assert risk.cvar_historico(r, a) >= risk.var_historico(r, a)
        assert risk.cvar_parametrico(r, a) >= risk.var_parametrico(r, a)


def test_var_parametrico_normal_aproxima_historico():
    r = np.random.default_rng(2).normal(0, 0.01, 200_000)
    assert risk.var_parametrico(r, 0.99) == pytest.approx(risk.var_historico(r, 0.99), rel=0.03)


def test_kupiec_nao_rejeita_taxa_correta_e_rejeita_excesso():
    assert risk.kupiec_pof(10, 1000, 0.99)[1] > 0.05
    assert risk.kupiec_pof(40, 1000, 0.99)[1] < 0.01


def test_christoffersen_detecta_agrupamento():
    iid = np.random.default_rng(3).binomial(1, 0.05, 2000)
    cluster = np.zeros(2000, dtype=int); cluster[100:140] = 1; cluster[800:840] = 1
    assert risk.christoffersen_independencia(iid)[1] > 0.01
    assert risk.christoffersen_independencia(cluster)[1] < 0.01


# ---- carteiras (Capítulos 2 e 4)
def test_pesos_somam_um_e_sem_venda_a_descoberto(mercado):
    mu, S = mercado
    for w in (portfolio.minima_variancia(S), portfolio.maximo_sharpe(mu, S, 0.05), portfolio.paridade_de_risco(S)):
        assert w.sum() == pytest.approx(1, abs=1e-8) and (w >= -1e-9).all()


def test_gmv_tem_menor_variancia_que_equiponderada(mercado):
    _, S = mercado
    assert portfolio.variancia_carteira(portfolio.minima_variancia(S), S) <= \
           portfolio.variancia_carteira(np.ones(3) / 3, S) + 1e-12


def test_tangente_tem_sharpe_maximo(mercado):
    mu, S = mercado
    sh = lambda w: (portfolio.retorno_carteira(w, mu) - 0.05) / np.sqrt(portfolio.variancia_carteira(w, S))
    assert sh(portfolio.maximo_sharpe(mu, S, 0.05)) >= sh(np.ones(3) / 3) - 1e-9


def test_paridade_de_risco_iguala_contribuicoes(mercado):
    _, S = mercado
    w = portfolio.paridade_de_risco(S); rc = w * (S @ w)
    assert rc.max() / rc.min() == pytest.approx(1, abs=1e-3)


def test_fronteira_e_crescente(mercado):
    mu, S = mercado
    r, v, _ = portfolio.fronteira_eficiente(mu, S, 20)
    assert (np.diff(v) >= -1e-9).all()


def test_beta_capm_recupera_beta():
    rng = np.random.default_rng(4); m = rng.normal(0.001, 0.01, 5000)
    a = 0.0002 + 1.3 * m + rng.normal(0, 0.002, 5000)
    assert portfolio.beta_capm(a, m)[1] == pytest.approx(1.3, abs=0.02)


# ---- renda fixa e valor do dinheiro no tempo (Capítulos 3 e 8)
def test_titulo_ao_par_quando_cupom_igual_ytm():
    assert fixed_income.preco_titulo(1000, 0.06, 0.06, 10) == pytest.approx(1000, abs=1e-9)


def test_ytm_inverte_preco():
    p = fixed_income.preco_titulo(1000, 0.06, 0.055, 10)
    assert fixed_income.ytm(p, 1000, 0.06, 10) == pytest.approx(0.055, abs=1e-9)


def test_duracao_aproxima_variacao_de_preco_e_convexidade_positiva():
    mac, mod = fixed_income.duracao(1000, 0.06, 0.055, 10)
    p0 = fixed_income.preco_titulo(1000, 0.06, 0.055, 10)
    p1 = fixed_income.preco_titulo(1000, 0.06, 0.056, 10)
    assert (p1 / p0 - 1) == pytest.approx(-mod * 0.001, rel=0.02)
    assert fixed_income.convexidade(1000, 0.06, 0.055, 10) > 0 and mac > mod


def test_tir_zera_vpl():
    f = [-100000, 30000, 35000, 40000, 45000, 50000]
    assert fixed_income.vpl(f, fixed_income.tir(f)) == pytest.approx(0, abs=1e-6)


def test_price_e_sac_quitam_a_divida_e_sac_paga_menos_juros():
    pr = fixed_income.tabela_price(300_000, 0.009, 360); sac = fixed_income.tabela_sac(300_000, 0.009, 360)
    assert pr[-1][4] == pytest.approx(0, abs=1e-6) and sac[-1][4] == pytest.approx(0, abs=1e-6)
    assert sum(l[2] for l in sac) < sum(l[2] for l in pr)


def test_taxa_forward_consistente_com_spot():
    f = fixed_income.taxa_forward(0.10, 1, 0.11, 2)
    assert (1.10) * (1 + f) == pytest.approx(1.11**2, rel=1e-12)


# ---- séries temporais (Capítulos 1, 10 e 18)
def test_retorno_log_aditivo_e_relacao_com_simples():
    p = np.array([100, 102, 99, 105.0])
    assert timeseries.retornos_log(p).sum() == pytest.approx(np.log(p[-1] / p[0]))
    assert np.allclose(np.exp(timeseries.retornos_log(p)) - 1, timeseries.retornos_simples(p))


def test_ljung_box_ruido_branco_vs_ar1():
    rng = np.random.default_rng(5); e = rng.normal(size=3000)
    ar = np.zeros(3000)
    for t in range(1, 3000): ar[t] = 0.6 * ar[t - 1] + e[t]
    assert timeseries.ljung_box(e, 10)[1] > 0.01 and timeseries.ljung_box(ar, 10)[1] < 1e-6


def test_garch_previsao_converge_para_variancia_de_longo_prazo():
    o, a, b = 0.05, 0.08, 0.9
    longo = o / (1 - a - b)
    assert timeseries.previsao_variancia_garch(o, a, b, 3 * longo, 2000) == pytest.approx(longo, rel=1e-6)
    assert timeseries.previsao_variancia_garch(o, a, b, 3 * longo, 1) == pytest.approx(3 * longo)


def test_drawdown_maximo_negativo():
    assert timeseries.drawdown_maximo(np.log([1.1, 0.9, 1.0, 0.8])) < 0
