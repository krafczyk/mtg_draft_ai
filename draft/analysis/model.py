import numpy as np
from numpy.typing import ArrayLike
import sympy
from draft.typing import Real, RealLike
from draft.utils import sympy_to_numpy
import itertools


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

        Probs: ArrayLike = np.array(probs)

        if np.sum(Probs) - 1. < 1e-6:
            Probs = Probs/np.sum(Probs)
        pool_id: str = np.random.choice(
            list(self.prob_map.keys()),
            p=Probs)
        return pool_id


class PackModel:
    def __init__(self, slot_definitions: list[SlotDefinition], card_pools: list[CardPool]):
        self.defs: list[SlotDefinition] = slot_definitions
        self.pools: list[CardPool] = card_pools
        self.unique_realizations: dict[str, tuple[list[str], list[list[str]]]] = {}

    def build_pool_realizations(self):
        slot_realizations: list[list[str]] = []

        # Build list of all realizations
        for I in itertools.product(*[ list(slot_def.prob_map.keys()) for slot_def in self.defs]):
            slot_realizations.append(list(I))

        # Build dict of unique realizations
        unique_realizations: dict[str, tuple[list[str], list[list[str]]]] = {}
        for s in slot_realizations:
            sorted_s = list(sorted(s))
            cat: str = ''.join(sorted_s)
            if cat not in unique_realizations:
                unique_realizations[cat] = (s, [s])
            else:
                unique_realizations[cat][1].append(s)

        def count_slots(realization: list[str]) -> dict[str,int]:
            counts: dict[str,int] = {}
            for slot in realization:
                if slot not in counts:
                    counts[slot] = 1
                else:
                    counts[slot] += 1
            return counts

        # Find slots common to all categories
        common_categories: dict[str, int] = count_slots(unique_realizations[list(unique_realizations.keys())[0]][0])

        for i in range(1, len(unique_realizations)):
            cat = list(unique_realizations.keys())[i]
            realization_count = count_slots(unique_realizations[cat][0])

            for k in list(common_categories.keys()):
                if k not in realization_count:
                    _ = common_categories.pop(k)
                elif realization_count[k] < common_categories[k]:
                    common_categories[k] = realization_count[k]

        # Take out common categories from key names
        for k in list(unique_realizations.keys()):
            # Get example realization
            ex_realization: list[str] = unique_realizations[k][0]
            # Take out common categories
            placeholder_realization: list[str] = []
            removed_count: dict[str, int] = {}
            for i in range(len(ex_realization)):
                cat = ex_realization[i]
                if cat not in removed_count:
                    removed_count[cat] = 0

                if cat in common_categories and common_categories[cat] > removed_count[cat]:
                    removed_count[cat] += 1
                else:
                    placeholder_realization.append(cat)

            new_k = ''.join(sorted(placeholder_realization))
            unique_realizations[new_k] = unique_realizations.pop(k)

        self.unique_realizations = unique_realizations

    def get_unique_sympy_symbols(self) -> set[sympy.Symbol]:
        temp_symbols: set[sympy.Basic] = set()
        for slot_def in self.defs:
            for _, v in slot_def.prob_map.items():
                if isinstance(v, sympy.Basic):
                    # Get all unique symbols from sympy expressions
                    temp_symbols.update(v.free_symbols)
        unique_symbols: set[sympy.Symbol] = set()
        for s in temp_symbols:
            if isinstance(s, sympy.Symbol):
                unique_symbols.add(s)
        return unique_symbols

    #def measure_unique_categories(self, pack_df):


    def fit_v1(self, pack_df):
        self.build_pool_realizations()

        # First we fit p_s

        ## Build the probability of a pack containing an spg
        p = []
        for cat, (_, realizations) in self.unique_realizations.items():
            if 's' in cat:
                p_all = []
                for realization in realizations:
                    p_realization = []
                    for i in range(len(realization)):
                        p_realization.append(self.defs[i].prob_map[realization[i]])
                    p_realization = sympy.Mul(*p_realization)
                    p_all.append(p_realization)
                p_all = sympy.Add(*p_all)
                p.append(p_all)
        p = sympy.simplify(sympy.Add(*p))



        return p




