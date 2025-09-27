from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from draft.booster.booster import BoosterModel


class BoosterGenBase(ABC):
    model: BoosterModel

    def __init__(self, model: BoosterModel):
        self.model = model

    @abstractmethod
    def sample(self, n_packs: int | np.int32 | np.int64=1) -> pd.DataFrame:
        # Check that all slots are sampleable
        for slot in self.model.slots:
            if not slot.is_sampleable():
                raise ValueError(f"Slot {slot.name} isn't sampleable!")
