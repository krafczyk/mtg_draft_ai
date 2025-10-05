from draft.datasets.seventeenlands import get_pack_data_summary_dense, get_card_df
from draft.datasets.ops import attach_card_metadata
from draft.booster.basic import PrintSheet, BoosterSlot
from draft.booster.booster import BoosterModel
import draft.analysis.set_solver as set_solver
import pandas as pd
import numpy as np
import sympy as sp
from scipy.special import logsumexp as _lse
from sympy.utilities.lambdify import lambdastr
from scipy.optimize import minimize
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as jnn
import jax
from pprint import pprint
idx = pd.IndexSlice


def main():
    # MKM pack data
    MKM_df = get_pack_data_summary_dense("pack_data/pack_data_MKM.csv")

    # Build print sheets
    card_df = get_card_df("cards.csv")

    MKM_df = attach_card_metadata(MKM_df, card_df, meta=('expansion', 'rarity', 'types'),priority_sets=['MKM', 'OTP', 'SPG'], index='columns')

    land_sel = MKM_df.columns.get_level_values('types').str.contains('Land')
    basic_sel = MKM_df.columns.get_level_values('types').str.contains('Basic')

    cols = MKM_df.columns
    common_cards = cols[cols.get_locs(idx[:,'MKM', 'common'])]
    uncommon_cards = cols[cols.get_locs(idx[:,'MKM', 'uncommon'])]
    rare_cards = cols[cols.get_locs(idx[:,'MKM', 'rare'])]
    rare_sl_cards = cols[cols.get_locs(idx[:,'MKM', 'rare', land_sel])]
    rare_nonsl_cards = cols[cols.get_locs(idx[:,'MKM', 'rare', ~land_sel])]
    mythic_cards = cols[cols.get_locs(idx[:,'MKM', 'mythic'])]
    remainder_cards = cols[cols.get_locs(idx[:,:,:,~basic_sel])].difference(common_cards.union(uncommon_cards).union(rare_cards).union(mythic_cards))

    rem_rare_cards = remainder_cards[remainder_cards.get_locs(idx[:,:,('rare','mythic')])].get_level_values('name')
    rem_common_cards = remainder_cards[remainder_cards.get_locs(idx[:,:,('common','uncommon')])].get_level_values('name')

    card_sets = {
        'c': common_cards.get_level_values('name'),
        'u': uncommon_cards.get_level_values('name'),
        'r1': rare_nonsl_cards.get_level_values('name'),
        'r2': rare_sl_cards.get_level_values('name'),
        'm': mythic_cards.get_level_values('name'),
        'rem': remainder_cards.get_level_values('name')}

    print([(f"{k}: {len(v)}") for k,v in card_sets.items()])

    # Build category counts Ks for each sheet.
    Ks = {k: MKM_df.loc[:len(MKM_df)/10,v].sum(axis=1) for k,v in card_sets.items()}
    Ks = pd.DataFrame({
        k: v
        for k, v in Ks.items()})
    Ks = Ks.value_counts()

    print_sheets = {
        'c': PrintSheet('c', {1: card_sets['c']}),
        'u': PrintSheet('u', {1: card_sets['u']}),
        'r1': PrintSheet('r1', {1: card_sets['r1']}),
        'r2': PrintSheet('r2', {1: card_sets['r2']}),
        'm': PrintSheet('m', {1: card_sets['m']}),
        'rem': PrintSheet('rem', {1: rem_rare_cards, 2: rem_common_cards}),
    }

    sheet_keys = ['c', 'u', 'r1', 'r2', 'm', 'rem']

    slots = {
        'c': ('c',),
        'cg': ('c', 'rem'),
        'u': ('u',),
        'rm': ('r1', 'm'),
        'w1': ('c', 'u', 'r1', 'r2', 'm'),
        'w2': ('c', 'u', 'r1', 'r2', 'm') }

    booster_spec = [
        'c', 'c', 'c', 'c', 'c', 'c',
        'cg',
        'u', 'u', 'u',
        'rm', 'w1', 'w2'
    ]

    # cpu_device = jax.devices('cpu')[0]
    # with jax.default_device(cpu_device):
    #     result = set_solver.fit_v1_nll_jax(Ks, sheet_keys, slots, booster_spec)
    #     ic(result)

    # result = set_solver.fit_v2_nll_sympy_jax(Ks, sheet_keys, slots, booster_spec)

    # result = set_solver.fit_v4_nll_sympy_jax(Ks, sheet_keys, slots, booster_spec)

    result = set_solver.fit_v5_nll_sympy_numpy(Ks, sheet_keys, slots, booster_spec)

    print("Parameter Fit Results:")
    pprint(result)


if __name__ == "__main__":
    from mk_ic import install
    install()
    main()
