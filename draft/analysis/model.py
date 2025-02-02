import numpy as np
from draft.typing import RealLike


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
