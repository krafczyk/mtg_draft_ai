import numpy as np
from numpy.typing import ArrayLike
import sympy
from draft.typing import Real, RealLike
from draft.utils import sympy_to_numpy, is_contained
import itertools
from dataclasses import dataclass, field


@dataclass
class CardPool:
    _id: str
    cards: list[int] | list[str] = field(default_factory=list)

    def sample(self) -> int | str:
        card: int | str = np.random.choice(self.cards)
        return card


class SlotDefinition(dict[str,RealLike]):

    def sample(self, sym_subs: None | list[tuple[sympy.Symbol,Real]] = None) -> str:
        probs: list[np.float32] = []
        for v in self.values():
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
            list(self.keys()),
            p=Probs)
        return pool_id


@dataclass
class CardGroupInfo:
    n: sympy.Symbol


SlotRealization = list[str]


@dataclass
class PackModel:
    defs: list[SlotDefinition] = field(default_factory=list)
    pools: dict[str, CardPool] = field(default_factory=dict)
    unique_realizations: dict[str, list[SlotRealization]] = field(default_factory=dict)
    master_realizations: dict[str, list[SlotRealization]] = field(default_factory=dict)
    master_unique_realizations: dict[str, list[SlotRealization]] = field(default_factory=dict)
    common_categories: dict[str, int] = field(default_factory=dict)
    pool_groups: list[SlotRealization] = field(default_factory=list)

    def build_pool_realizations(self):
        slot_realizations: list[SlotRealization] = []

        # Build list of all realizations
        for I in itertools.product(*[ list(slot_def.prob_map.keys()) for slot_def in self.defs]):
            slot_realizations.append(list(I))

        # Build dict of unique realizations
        unique_realizations: dict[str, tuple[SlotRealization, list[SlotRealization]]] = {}
        for s in slot_realizations:
            sorted_s = list(sorted(s))
            cat: str = ''.join(sorted_s)
            if cat not in unique_realizations:
                unique_realizations[cat] = (s, [s])
            else:
                unique_realizations[cat][1].append(s)

        def count_slots(realization: SlotRealization) -> dict[str,int]:
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
            ex_realization: SlotRealization = unique_realizations[k][0]
            # Take out common categories
            placeholder_realization: SlotRealization = []
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

        for k in list(unique_realizations.keys()):
            self.unique_realizations[k] = unique_realizations[k][1]
        self.common_categories = common_categories

        # Some categories are indistinguishable, Let's detect them
        self.pool_groups = []
        pool_keys = list(self.pools.keys())
        for i in range(len(pool_keys)):
            p_main_id = pool_keys[i]
            # See if this pool is contained or contains any of the pools
            # already encountered
            found_existing_group = False
            for j in range(len(self.pool_groups)):
                join_group = False
                for p_sec_id in self.pool_groups[j]:
                    l_1 = self.pools[p_main_id].cards
                    l_2 = self.pools[p_sec_id].cards
                    if is_contained(l_1, l_2) or is_contained(l_2, l_1):
                        join_group = True
                        break
                if join_group:
                    self.pool_groups[j].append(p_main_id)
                    found_existing_group = True
                    break
            if not found_existing_group:
                self.pool_groups.append([p_main_id])

        # For each group, detect the 'master' group. The one which contains the others.
        master_pools = []
        for i in range(len(self.pool_groups)):
            for p_id in self.pool_groups[i]:
                master = True
                for j in range(len(self.pool_groups)):
                    if i == j:
                        continue
                    for p_sec_id in self.pool_groups[j]:
                        l_1 = self.pools[p_id].cards
                        l_2 = self.pools[p_sec_id].cards
                        if is_contained(l_1, l_2):
                            master = False
                            break
                    if not master:
                        break
                if master:
                    master_pools.append(p_id)
                    break
            if len(master_pools) < i+1:
                master_pools.append(None)

        if None in master_pools:
            raise ValueError("Master pool not found for some pool groups, this is currently unsupported")

        # Sanity check
        assert len(master_pools) == len(self.pool_groups)

        # capitalize the master pools with multiple sub pools to distinguish them
        for i in range(len(self.pool_groups)):
            if len(self.pool_groups[i]) > 1:
                master_pools[i] = str.upper(master_pools[i])

        master_pool_map: dict[str,str] = {}
        for i in range(len(self.pool_groups)):
            for p_id in self.pool_groups[i]:
                master_pool_map[p_id] = master_pools[i]

        master_common_categories = {}
        for k in self.common_categories:
            master_common_categories[master_pool_map[k]] = self.common_categories[k]

        self.master_realizations = {}
        self.master_unique_realizations = {}
        for r_key in self.unique_realizations:
            for r in self.unique_realizations[r_key]:
                m_r = [ master_pool_map[p] for p in r ]
                key_count: dict[str,int] = {}
                for k in m_r:
                    key_count[k] = key_count.get(k, 0) + 1
                for k in master_common_categories:
                    key_count[k] = key_count[k]-master_common_categories[k]
                m_r_temp: SlotRealization = []
                for k in key_count:
                    m_r_temp += [k]*key_count[k]
                m_r_key: str = ''.join(sorted(m_r_temp))
                # We assume these are unique realizations so we don't need to check if
                # we've seen a particular realization before
                if m_r_key in self.master_unique_realizations:
                    self.master_unique_realizations[m_r_key].append(r)
                else:
                    self.master_unique_realizations[m_r_key] = [r]
                # Here we do need to check if we've seen a particular realization before
                if m_r_key in self.master_realizations:
                    if m_r not in self.master_realizations[m_r_key]:
                        self.master_realizations[m_r_key].append(m_r)
                else:
                    self.master_realizations[m_r_key] = [m_r]

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
        for cat, realizations in self.unique_realizations.items():
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




