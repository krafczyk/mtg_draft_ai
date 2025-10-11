import dask
import sympy
import numpy as np
import pandas as pd
from typing import Any, TypeVar
import numbers
from sympy import Symbol, sympify, N
import numpy as np
from collections.abc import Mapping, Sequence


def configure_dask():
    _ = dask.config.set(scheduler='processes')


def sympy_to_numpy(x: sympy.Basic, precision: str = 'high') -> None | np.integer[Any] | np.floating[Any]:
    if type(x) is sympy.Float:
        if precision == 'high':
            return np.float64(x)
        else:
            return np.float32(x)
    elif type(x) is sympy.Integer:
        if precision == 'high':
            return np.int64(x)
        else:
            return np.int32(x)
    elif type(x) is sympy.Expr:
        if precision == 'high':
            return np.float64(x)
        else:
            return np.float32(x)
    else:
        raise TypeError(f"Unsupported type {type(x)}")


T = TypeVar('T')  # any type supporting equality (all types in Python do)

def is_contained(sub: list[T], main: list[T]) -> bool:
    # implementation here
    for el in sub:
        if el not in main:
            return False
    return True


def eval_real(expr: Any, values: dict[Any, np.float32]) -> np.float32:
    """
    Evaluate `expr` to a real (float) using `values` for any SymPy symbols.
    - If `expr` is already a Python/NumPy real number, returns it as float.
    - If `expr` is a SymPy Symbol, looks it up in `values`.
    - If `expr` is a SymPy expression, substitutes from `values` and evaluates.
    - Keys in `values` can be SymPy Symbols or their names (strings).
    - Raises KeyError if required symbols are missing.
      otherwise raises ValueError.
    """
    # Fast path for plain numbers
    if isinstance(expr, numbers.Real):
        return np.float32(expr)

    # Convert to a SymPy expression
    e = sympify(expr)

    # Build a substitution map accepting both Symbol and string keys
    subs_map = {}
    for k, v in values.items():
        s = k if isinstance(k, Symbol) else Symbol(str(k))
        subs_map[s] = v

    # Substitute if there are symbols; otherwise leave as-is
    e2 = e.subs(subs_map) if e.free_symbols else e

    # Check for missing symbols
    if e2.free_symbols:
        missing = ", ".join(sorted(s.name for s in e2.free_symbols))
        raise KeyError(f"Missing values for symbols: {missing}")

    # Numeric evaluation
    val = N(e2)

    # Handle real/complex policy
    if val.is_real is False:
        raise ValueError(f"Expression evaluated to non-real value: {val}")

    return np.float32(val)


def sort_card_df(df: pd.DataFrame,
               name_col='name',
               expansion_col='expansion',
               priority=('MKM', 'OTP', 'SPG')) -> pd.DataFrame:
    # Build categories: priority first, then the rest in lexicographic order
    present = pd.Index(df[expansion_col].dropna().unique())
    cats = list(priority) + sorted(x for x in present if x not in priority)

    exp_type = pd.CategoricalDtype(categories=cats, ordered=True)
    out = df.copy()
    out[expansion_col] = out[expansion_col].astype(exp_type)
    return out.sort_values([name_col, expansion_col], kind="mergesort")


def sp_evaluate(obj, subs=None, dps=None, chop=False, evalf=True):
    """
    Recursively evaluate SymPy objects inside nested containers.

    - subs: dict of substitutions (Symbols -> values/expressions)
    - dps:  precision for evalf/N (e.g., 50)
    - chop: zero out tiny residuals
    - evalf: if False, only apply .subs()
    """
    subs = subs or {}

    def _is_container(x):
        return isinstance(x, (Mapping, Sequence, set, np.ndarray)) and not isinstance(x, (str, bytes))

    def _eval_atom(x):
        # Anything with .subs/.evalf (Expr, Matrix, Indexed, etc.)
        if hasattr(x, "subs") and hasattr(x, "evalf"):
            y = x.subs(subs) if subs else x
            return (y.evalf(n=dps, chop=chop) if evalf else y)
        return x  # plain Python number, string, etc.

    def _eval(x):
        if isinstance(x, Mapping):
            return type(x)({ _eval(k): _eval(v) for k, v in x.items() })
        if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
            return type(x)(*(_eval(v) for v in x))
        if isinstance(x, (list, tuple)):
            return type(x)(_eval(v) for v in x)
        if isinstance(x, set):
            return { _eval(v) for v in x }
        if isinstance(x, np.ndarray):
            return np.vectorize(_eval, otypes=[object])(x)
        return _eval_atom(x)

    return _eval(obj)
