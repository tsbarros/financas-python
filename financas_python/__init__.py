"""financas_python: biblioteca de apoio ao livro "Finanças com Python: Da Teoria à Aplicação".

Uso:
    from financas_python import portfolio, risk, options, fixed_income, timeseries
"""
from . import portfolio, risk, options, fixed_income, timeseries

__version__ = "0.1.0"
__all__ = ["portfolio", "risk", "options", "fixed_income", "timeseries"]
