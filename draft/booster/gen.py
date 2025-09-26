from abc import ABC, abstractmethod
import numpy as np
from draft.booster.basic import BoosterModel


class BoosterGenBase(ABC):
    model: BoosterModel

    def __init__(self, model: BoosterModel):
        self.model = model

    @abstractmethod
    def sample(self, n_packs: int | np.int32 | np.int64=1):
        # Check that all slots are sampleable
        for slot in self.model.slots:
            if not slot.is_sampleable():
                raise ValueError(f"Slot {slot.name} isn't sampleable!")
