from draft.datasets.seventeenlands import get_pack_data_summary_dense, get_card_df
from draft.datasets.ops import attach_card_metadata
from draft.booster.basic import PrintSheet, BoosterSlot
from draft.booster.booster import BoosterModel
import pandas as pd
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdastr
from scipy.optimize import minimize
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as jnn
import jax
idx = pd.IndexSlice


def fit_nll_sympy_jax(Ks, sheet_keys, slots, booster_spec):
    # Validate booster_spec
    slot_ids = set(slots.keys())
    for slot in booster_spec:
        if slot not in slot_ids:
            raise ValueError(f"Booster spec contains slot '{slot}' not found in slots: {slot_ids}")

    # Validate slots
    for slot_name, slot_sheets in slots.items():
        for sheet in slot_sheets:
            if sheet not in sheet_keys:
                raise ValueError(f"Slot '{slot_name}' contains sheet '{sheet}' not found in sheets: {list(sheet_keys.keys())}")

    # Validate Ks
    if set(Ks.index.names) != set(sheet_keys):
        raise ValueError(f"Ks index names {Ks.index.names} do not match sheets {sheet_keys}")

    L = len(booster_spec)
    S = len(sheet_keys)
    N = len(Ks)

    # Extract numpy arrays from Ks and validate their shapes
    K_np = Ks.index.to_frame(index=False).to_numpy(dtype=np.int32)
    N_np = Ks.to_numpy(dtype=np.float32)
    assert K_np.shape == (N, S), f"Expected K shape {(N,S)}, got {K_np.shape}"
    assert N_np.shape == (N,), f"Expected N shape {(N,)}, got {N_np.shape}"
    ic(K_np.shape, N_np.shape)

    # Create z variable for each sheet
    z_vars = {sheet: sp.Symbol(f"z_{sheet}", real=True, positive=True) for sheet in sheet_keys}

    # Create slot prob expressions
    slot_probs = {}
    prob_vector = []
    for slot_key, slot_sheets in slots.items():
        if len(slot_sheets) == 1:
            slot_probs[slot_key] = {slot_sheets[0]: 1.}
        else:
            slot_probs[slot_key] = {}
            for sheet in slot_sheets:
                p = sp.Symbol(f"p_{slot_key}_{sheet}", real=True, positive=True)
                prob_vector.append(p)
                slot_probs[slot_key][sheet] = p

    ic(slot_probs, prob_vector)

    # Build PGF
    g = 1.

    g = sp.prod([
        sum([
            slot_probs[slot_key].get(sheet,0.) * z_vars[sheet]
            for sheet in slot_probs[slot_key]
        ])
        for slot_key in booster_spec
    ])

    gz = g.expand()

    # Build negative log-likelihood
    log_likelihood_terms = []
    for k, n in zip(K_np, N_np):
        z = sp.prod([z_vars[sheet]**ki for sheet, ki in zip(sheet_keys, k)])
        prob_term = gz.coeff(z)
        if prob_term == 0:
            continue
        log_likelihood_terms.append(n*sp.log(prob_term))
    log_likelihood = sp.Add(*log_likelihood_terms)

    nll_sp = -log_likelihood

    assert len(nll_sp.free_symbols.difference(set(prob_vector))) == 0

    nll_np_fn_1 = sp.lambdify(prob_vector, nll_sp, 'jax')

    # pre compute slot logit extraction indices
    logit_idx = {}
    softmax_slice_starts = []
    softmax_slice_sizes = []
    num_pars = 0
    for slot_key, slot_sheets in slots.items():
        ic(slot_key, slot_sheets)
        num_sheets = len(slot_sheets)
        if num_sheets > 1:
            softmax_slice_starts.append(num_pars)
            softmax_slice_sizes.append(num_sheets)
            for sheet in slot_sheets:
                logit_idx[(slot_key,sheet)] = num_pars
                num_pars += 1
    print(f"There are {num_pars} free parameters")

    # Prepare jax arrays
    softmax_slice_starts = tuple(softmax_slice_starts)
    softmax_slice_sizes = tuple(softmax_slice_sizes)

    # Precompute scatter indices for prob placement, and place needed 1.s
    ic(sheet_keys)
    invert_slot_key = []
    invert_sheet_key = []
    slot_i = 0
    for i, slot_key in enumerate(booster_spec):
        ic(i, slot_key)
        slot_sheets = slots[slot_key]
        if len(slot_sheets) > 1:
            for sheet_key in slot_sheets:
                j = sheet_keys.index(sheet_key)
                invert_slot_key.append(slot_key)
                invert_sheet_key.append(sheet_key)
                slot_i += 1

    @jax.jit
    def logits_to_probs(x):
        return jnp.concatenate([
            jnn.softmax(lax.dynamic_slice(x, (start,), (size,)))
            for start, size in zip(softmax_slice_starts, softmax_slice_sizes)
        ], axis=0)

    # Build NLL function
    @jax.jit
    def nll_fn_logits(x):
        # Input x is a flat packing of logits for each slot with multiple sheets        

        # Compute/update x slices with softmax
        return nll_np_fn_1(*logits_to_probs(x))

    x0 = np.random.random(size=(num_pars,))

    # value + gradient in one JIT-compiled call
    val_and_grad = jax.jit(jax.value_and_grad(nll_fn_logits))

    # SciPy expects (f, g) with NumPy types when jac=True
    def scipy_obj(x_np):
        x = jnp.asarray(x_np)
        f, g = val_and_grad(x)
        return float(f), np.asarray(g)

    # Warm-up compile (optional, avoids first-call compile during minimize)
    _ = scipy_obj(np.asarray(x0, dtype=np.float64))

    res = minimize(scipy_obj,
                   x0=np.asarray(x0, dtype=np.float64),
                   method="L-BFGS-B",
                   jac=True)

    ic(res)

    x_fit = res.x

    p_fit = logits_to_probs(x_fit)

    # 1) Compute hessian at solution
    hess_fn = jax.jit(jax.hessian(nll_fn_logits))
    H = np.asarray(hess_fn(x_fit))

    # 2) Invert hessian to get covariance of logits
    try:
        cov_x = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_x = np.linalg.pinv(H) # common if using full softmax blocks

    # 3) Jacobian of probs wrt logits at x_fit
    Jp_fn = jax.jit(jax.jacrev(logits_to_probs))
    Jp = np.asarray(Jp_fn(x_fit))

    # 4) Delta method: Cov of probabilities and SEs
    cov_p = Jp @ cov_x @ Jp.T

    # guard tiny negative due to numerics
    var_p = np.clip(np.diag(cov_p), 0.0, None)
    se_p = np.sqrt(var_p)

    z = 1.96

    # map vector back to slot/sheet dict
    result = {}
    for i, (slot_key, sheet_key) in enumerate(zip(invert_slot_key, invert_sheet_key)):
        if slot_key not in result:
            result[slot_key] = {}
        result[slot_key][sheet_key] = (
            np.float32(p_fit[i]), np.float32(z*se_p[i]))

    ic(result)


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
    Ks = {k: MKM_df.loc[:,v].sum(axis=1) for k,v in card_sets.items()}
    Ks = pd.DataFrame({
        k: v
        for k, v in Ks.items()})
    Ks = Ks.value_counts()
    ic(Ks)

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
        'c+g': ('c', 'rem'),
        'u': ('u',),
        'rm': ('r1', 'm'),
        'w1': ('c', 'u', 'r1', 'r2', 'm'),
        'w2': ('c', 'u', 'r1', 'r2', 'm') }

    booster_spec = [
        'c', 'c', 'c', 'c', 'c', 'c',
        'c+g',
        'u', 'u', 'u',
        'rm', 'w1', 'w2'
    ]

    fit_nll_sympy_jax(Ks, sheet_keys, slots, booster_spec)


if __name__ == "__main__":
    from mk_ic import install
    install()
    main()
