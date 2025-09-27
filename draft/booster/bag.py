import pandas as pd
import numpy as np
from typing import TypeVar

from draft.booster.basic import PrintSheet
from draft.booster.booster import BoosterModel
from draft.booster.gen import BoosterGenBase 


class Bag:
    def __init__(self, print_sheet:PrintSheet):
        self.master_sheet: list[str] = []
        for card, count in print_sheet.cards.items():
            self.master_sheet.extend([card.name] * count)
        self.refill()

    def refill(self):
        # Shuffle the bag
        self.content: list[str] = self.master_sheet.copy()

    def sample(self) -> str:
        # Randomly grab one from the bag
        val = self.content.pop(np.random.randint(len(self.content)))
        if len(self.content) == 0:
            self.refill()
        return val


class BagGen1(BoosterGenBase):
    def __init__(self, model: BoosterModel, num_bags:int = 1, skip:float=0.):
        super().__init__(model)

        # Create bags for each unique print sheet/slot combination
        self.cards = self.model.get_card_name_list()
        self.num_bags = num_bags
        self.skip = skip
        self.sheet_bags: dict[tuple[str,str],list[Bag]] = {}
        for slot in model.slots:
            for sheet_spec in slot.sheets:
                new_bag_list = []
                for _ in range(num_bags):
                    new_bag_list.append(Bag(sheet_spec.sheet))
                self.sheet_bags[slot.name, sheet_spec.sheet.name] = new_bag_list

    def sample(self, n_packs: int | np.int32 | np.int64=1) -> pd.DataFrame:
        super().sample(n_packs)
        # Create array to hold pack information
        pack_data = np.zeros((n_packs, len(self.cards)), dtype=np.uint8)
        packs_sampled = 0
        while packs_sampled < n_packs:
            card_list = []
            for slot in self.model.slots:
                sheet = slot.sample_sheet()
                bag_i = np.random.choice(np.arange(self.num_bags))
                card_name = self.sheet_bags[slot.name, sheet.name][bag_i].sample()
                card_list.append(card_name)

            if np.random.random() < self.skip:
                continue
            for card_name in card_list:
                card_idx = self.cards.index(card_name)
                pack_data[packs_sampled,card_idx] += 1
            packs_sampled += 1

        return pd.DataFrame(pack_data, columns=self.cards)
