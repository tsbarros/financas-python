# financas_python

Biblioteca de apoio ao livro *Finanças com Python: Da Teoria à Aplicação* (Thiago de Sousa Barros).

```bash
pip install -e .[teste]
python -m pytest -q
```

```python
from financas_python import portfolio, risk, options, fixed_income, timeseries
```

| Módulo | Conteúdo | Capítulos |
|---|---|---|
| `portfolio` | GMV, tangente, fronteira, paridade de risco, beta CAPM | 2, 4, 5 |
| `risk` | VaR/CVaR (histórico e paramétrico), Kupiec, Christoffersen | 7 |
| `options` | Binomial CRR, Black-Scholes-Merton, gregas, vol. implícita | 16 |
| `fixed_income` | VP/VPL/TIR, Price/SAC, preço, duração, convexidade, YTM, DI | 3, 8, 15 |
| `timeseries` | retornos, estatísticas, Ljung-Box, GARCH(1,1) | 1, 10, 18 |

Convenções: retornos em decimal; VaR e CVaR são perdas positivas; taxas em decimal a.a., exceto `pu_di` (em %).
Dependências: apenas `numpy` e `scipy`. As funções de acesso a dados (yfinance, BCB) permanecem nos capítulos.
