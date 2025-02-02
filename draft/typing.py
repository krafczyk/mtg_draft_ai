import numpy as np
import sympy
from typing import Any

RealLike = int | float | np.integer[Any] | np.floating[Any] | sympy.Basic # pyright: ignore[reportExplicitAny]
