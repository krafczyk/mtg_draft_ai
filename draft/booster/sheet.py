from abc import ABC, abstractmethod
import numpy as np


class SheetGenBase(ABC):
    @abstractmethod
    def sample(self, n_packs: int | np.int32 | np.int64=1): ...
