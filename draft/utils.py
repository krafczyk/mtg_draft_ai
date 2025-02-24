import dask
import sympy
import numpy as np
from typing import Any, TypeVar


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
