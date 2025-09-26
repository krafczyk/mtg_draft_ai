import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from draft.typing import RealLike
from typing import cast
import sympy as sp
from draft.utils import eval_real


class Card:
    name: str
    set_code: str
    collector_number: int

    def __init__(self, name:str, set_code:str|None=None, collector_number:int|None=None):
        self.name = name
        if set_code is None:
            self.set_code = ""
        if collector_number is None:
            self.collector_number = 0

    def __hash__(self):
        return hash(f"{self.name}_{self.set_code}_{self.collector_number}")


@dataclass
class PrintSheet:
    name: str
    cards: dict[Card, int]
    card_names: NDArray[np.str_]
    card_idxs: NDArray[np.int32]
    card_weights: NDArray[np.float32]

    def __init__(self, name:str , cards:dict[str|Card,int]|list[str|Card]):
        self.name = name
        # Turn cards into a dict if it's a list
        if isinstance(cards, list):
            cards = {
                Card(name=card) if isinstance(card, str) else card: 1 for card in cards
            }
        if isinstance(cards, dict):
            cards = {
                Card(name=card) if isinstance(card, str) else card: count for card, count in cards.items()
            }
        self.cards = cast(dict[Card,int], cards)
        self.card_idxs = np.arange(len(cards), dtype=np.int32)
        self.card_weights = np.array(list(cards.values()), dtype=np.float32)
        self.card_weights = self.card_weights / np.sum(self.card_weights)
        self.card_names = np.array([card.name for card in self.cards.keys()], dtype=np.str_)

    def __len__(self):
        return len(self.cards)


@dataclass
class SheetSpec:
    prob: RealLike | None
    sheet: PrintSheet


class BoosterSlot:
    name: str
    sheets: list[SheetSpec]
    cumulative_probs: NDArray[np.float32] | None

    def __init__(self, name:str, sheets: list[SheetSpec|PrintSheet]):
        self.name = name
        self.sheets = []
        for sheet in sheets:
            if isinstance(sheet, PrintSheet):
                self.sheets.append(SheetSpec(prob=None, sheet=sheet))
            else:
                self.sheets.append(sheet)
        n_none = sum(1 for sheet in self.sheets if sheet.prob is None)

        if n_none == 0:
            raise ValueError("At least one sheet must have unspecified probability")
        if n_none > 1:
            raise ValueError("At most one sheet can have unspecified probability")
        total_prob = sum(sheet.prob for sheet in self.sheets if sheet.prob is not None)
        i_none = next(i for i, sheet in enumerate(self.sheets) if sheet.prob is None)
        self.sheets[i_none].prob = 1 - total_prob

        try:
            self.set_slot_probs()
        except TypeError:
            pass


    def is_sampleable(self) -> bool:
        if self.cumulative_probs is None:
            return False
        else:
            return True

    def sample_sheet(self) -> PrintSheet:
        v = np.random.random()
        sheet_idx = np.searchsorted(self.cumulative_probs, v, side='right')
        return self.sheets[sheet_idx].sheet

    def set_slot_probs(self, prob_dict: dict[str|sp.Symbol,RealLike]|None=None):
        if prob_dict is None:
            try:
                sheet_probs = [np.float32(sheet.prob) for sheet in self.sheets]
                self.cumulative_probs = np.cumsum(sheet_probs, dtype=np.float32)
            except Exception as e:
                self.cumulative_probs = None
                raise e
        else:
            try:
                sheet_probs = [eval_real(sheet.prob, prob_dict) for sheet in self.sheets]
                self.cumulative_probs = np.cumsum(sheet_probs, dtype=np.float32)
            except Exception as e:
                self.cumulative_probs = None
                raise e


class BoosterModel:
    slots: list[BoosterSlot]

    def __init__(self, slots: list[BoosterSlot]):
        self.slots = slots
        # Check that all sheets are mutually exclusive
        unique_sheets = self.get_unique_sheets()
        for i in range(len(unique_sheets)):
            for j in range(i + 1, len(unique_sheets)):
                sheet_i = unique_sheets[i]
                sheet_j = unique_sheets[j]
                names_i = set(sheet_i.cards.keys())
                names_j = set(sheet_j.cards.keys())
                intersection = names_i.intersection(names_j)
                if intersection:
                    raise ValueError(f"Sheets {sheet_i.name} and {sheet_j.name} share the following cards: {intersection}")

    def get_card_name_list(self) -> list[str]:
        unique_sheets = self.get_unique_sheets()
        all_cards = []
        for sheet in unique_sheets:
            all_cards.extend(list(map(lambda c: c.name, sheet.cards.keys())))

        all_cards.sort()
        return all_cards

    def get_unique_sheets(self) -> list[PrintSheet]:
        seen = set()
        unique_sheets = []
        for slot in self.slots:
            for sheet_spec in slot.sheets:
                if sheet_spec.sheet.name not in seen:
                    seen.add(sheet_spec.sheet.name)
                    unique_sheets.append(sheet_spec.sheet)
        return unique_sheets

    def set_slot_probs(self, prob_dict: dict[str|sp.Symbol,RealLike]|None=None):
        for slot in self.slots:
            slot.set_slot_probs(prob_dict=prob_dict)
