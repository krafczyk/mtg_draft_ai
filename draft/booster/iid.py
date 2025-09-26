import pandas as pd
import numpy as np
from typing import TypeVar

from draft.booster.gen import BoosterGenBase 

class IIDGen(BoosterGenBase):
    def sample(self, n_packs: int | np.int32 | np.int64=1):
        cards = self.model.get_card_name_list()
        # Create array to hold pack information
        pack_data = np.zeros((n_packs, len(cards)), dtype=np.uint8)
        for pack_i in range(n_packs):
            pack = []
            for slot in self.model.slots:
                sheet = slot.sample_sheet()
                card_idx = np.random.choice(sheet.card_idxs, p=sheet.card_weights)
                card_name = sheet.card_names[card_idx]
                card_idx = cards.index[card_name]
                pack_data[pack_i,card_idx] += 1

        return pd.DataFrame(pack_data, columns=cards)
