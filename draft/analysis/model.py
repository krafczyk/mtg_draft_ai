import numpy as np
import sympy
from draft.typing import Real, RealLike
from draft.utils import sympy_to_numpy


class CardPool:
    def __init__(self, _id: str, cards: list[int] | list[str]):
        self._id: str = _id
        self.cards: list[int] | list[str] = cards

    def sample(self) -> int | str:
        card: int | str = np.random.choice(self.cards)
        return card


class SlotDefinition:
    def __init__(self, prob_map: dict[str, RealLike]):
        self.prob_map: dict[str, RealLike] = prob_map

    def sample(self, sym_subs: None | list[tuple[sympy.Symbol,Real]] = None) -> str:
        probs: list[np.float32] = []
        for v in self.prob_map.values():
            w: Real = 0.
            if isinstance(v, sympy.Basic):
                if sym_subs is not None:
                    w = np.float32(sympy_to_numpy(v.subs(sym_subs)))
            else:
                w = v
            probs.append(np.float32(w))

        pool_id: str = np.random.choice(
            list(
                self.prob_map.keys()),
            p=probs)
        return pool_id
