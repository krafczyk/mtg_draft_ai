import numpy as np
from typing import TypeVar


T = TypeVar('T')

def sample_pool(pool: list[T], p: None | list[float]=None) -> T:
    """
    Samples one of the pool elements. use probability weights p if specified
    """
    return np.random.choice(pool, p=p) # pyright: ignore[reportCallIssue,reportArgumentType]
