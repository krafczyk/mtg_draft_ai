from draft.datasets.seventeenlands import get_pack_data_summary_dense, get_card_df
from draft.datasets.ops import attach_card_metadata
from draft.booster.basic import PrintSheet, BoosterSlot
from draft.booster.booster import BoosterModel
import pandas as pd
import numpy as np
import sympy as sp
from scipy.optimize import minimize
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as jnn
import jax
idx = pd.IndexSlice


def _shift_pad_axis(x, ax:int):
    """Shift x by +1 along axis ax, pading a zero in front (static ax)."""
    shape = x.shape
    # slice sizes: full, except along ax where it's -1
    slice_sizes = list(shape)
    slice_sizes[ax] = shape[ax]-1
    # take leading slice of size...size-1 along ax
    x_slice = lax.dynamic_slice(x, start_indices=(0,)*x.ndim, slice_sizes=tuple(slice_sizes))

    # place it into zeros starting at index 1 on ax
    start = [0]*x.ndim
    start[ax] = 1
    out = jnp.zeros_like(x)
    return lax.dynamic_update_slice(out, x_slice, tuple(start))


def _all_axis_shifts(x):
    """Return stacked shifts for every axis: shape(S, *x.shape)."""
    return jnp.stack([_shift_pad_axis(x, ax) for ax in range(x.ndim)], axis=0)


# def pack_count_dist(slot_probs): # slot_probs shape (L,S): L -> number of slots S -> number of independent print sheets
#     L, S = slot_probs.shape

#     dp0 = jnp.zeros((L+1,)*S, dtype=slot_probs.dtype).at[(0,)*S].set(1.0)

#     def body(i, dp_prev):
#         acc = jnp.zeros_like(dp_prev)
#         def add_one(ax, acc_):
#             return acc_ + slot_probs[i, ax]* _shift_pad_axis(dp_prev, ax)
#         return lax.fori_loop(0, S, add_one, acc)
#     return lax.fori_loop(0, L, body, dp0)


def _shift_pad_axis_static(x, ax: int):
    """Shift x by +1 along axis `ax` (static), padding a zero in front."""
    shape = x.shape
    # slice: take all but the last element along `ax`
    x_slice = lax.slice_in_dim(x, start_index=0, limit_index=shape[ax]-1, stride=1, axis=ax)
    # write the slice into zeros starting at index 1 along `ax`
    start = [0] * x.ndim
    start[ax] = 1
    return lax.dynamic_update_slice(jnp.zeros_like(x), x_slice, tuple(start))

def _shift_pad_axis_switch(x, ax_traced):
    """Dispatch to a static-ax version using lax.switch."""
    branches = tuple(
        (lambda a: (lambda y: _shift_pad_axis_static(y, a)))(a)  # capture a
        for a in range(x.ndim)
    )
    return lax.switch(ax_traced, branches, x)

def pack_count_dist(slot_probs):
    """
    slot_probs: (L, S) routing probs per slot over S sheets.
    Returns dp: (L+1,)*S tensor of joint counts.
    """
    L, S = slot_probs.shape
    dp0 = jnp.zeros((L+1,)*S, dtype=slot_probs.dtype).at[(0,)*S].set(1.0)

    def body(i, dp_prev):
        # accumulate sum over all axes without stacking
        def add_one(ax, acc):
            shifted = _shift_pad_axis_switch(dp_prev, ax)      # axis decision is static per branch
            return acc + slot_probs[i, ax] * shifted
        return lax.fori_loop(0, S, add_one, jnp.zeros_like(dp_prev))

    return lax.fori_loop(0, L, body, dp0)


def loglikelihood_from_counts(dp, K, N, eps=1e-15):
    """
    dp: (L+1,)*S tensor of joint pack outcome probabilities
    K: (N, S) integer counts per per unique K, N unique ks
    N: (N,) weights (counts)
    """
    # Gather probabilities at each K row
    # tuple(K.T) is (S,) tuple of arrays, one per axis
    pi_at_K = dp[tuple(K.T)]  # shape (N,)
    ll = jnp.dot(N, jnp.log(jnp.clip(pi_at_K, eps, 1.0)))
    return ll

def fit_nll_jax(Ks, sheet_keys, slots, booster_spec):
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

    # Extract shape values that are constant for this problem
    L = len(booster_spec)  # number of slots in booster
    S = len(sheet_keys)    # number of independent print sheets
    N = len(Ks)            # number of unique pack outcomes

    print(f"Fitting NLL with L={L}, S={S}, N={N}")

    # Extract numpy arrays from Ks and validate their shapes
    K_np = Ks.index.to_frame(index=False).to_numpy(dtype=np.int32)
    N_np = Ks.to_numpy(dtype=np.float32)
    assert K_np.shape == (N, S), f"Expected K shape {(N,S)}, got {K_np.shape}"
    assert N_np.shape == (N,), f"Expected N shape {(N,)}, got {N_np.shape}"
    ic(K_np.shape, N_np.shape)

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
    #softmax_slice_starts = jnp.array(softmax_slice_starts, dtype=jnp.int32)
    softmax_slice_starts = tuple(softmax_slice_starts)
    #softmax_slice_sizes = jnp.array(softmax_slice_sizes, dtype=jnp.int32)
    softmax_slice_sizes = tuple(softmax_slice_sizes)
    ic(logit_idx, softmax_slice_starts, softmax_slice_sizes)

    # Precompute initial slot prob structure
    init_slot_probs = np.zeros((L, S), dtype=np.float32)

    # Precompute scatter indices for prob placement, and place needed 1.s
    ic(sheet_keys)
    out_rows, out_cols = [], []
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
                out_rows.append(i)
                out_cols.append(j)
                slot_i += 1
        else:
            sheet_key = slot_sheets[0]
            j = sheet_keys.index(sheet_key)
            init_slot_probs[i, j] = 1.

    # Prepare jax arrays
    init_slot_probs = jnp.array(init_slot_probs)
    out_rows = tuple(out_rows)
    out_cols = tuple(out_cols)

    K_jnp = jnp.array(K_np)
    N_jnp = jnp.array(N_np)

    ic(init_slot_probs, out_rows, out_cols)

    @jax.jit
    def logits_to_probs(x):
        return jnp.concatenate([
            jnn.softmax(lax.dynamic_slice(x, (start,), (size,)))
            for start, size in zip(softmax_slice_starts, softmax_slice_sizes)
        ], axis=0)

    @jax.jit
    def nll_fn(p):
        # Input x is a flat packing of logits for each slot with multiple sheets        

        # Scatter to full slot prob array
        slot_probs = init_slot_probs.at[(out_rows, out_cols)].set(p)

        pack_count_probs = pack_count_dist(slot_probs) # shape (L+1, S)

        return -loglikelihood_from_counts(pack_count_probs, K_jnp, N_jnp)

    # Build NLL function
    @jax.jit
    def nll_fn_logits(x):
        # Input x is a flat packing of logits for each slot with multiple sheets        

        # Compute/update x slices with softmax
        return nll_fn(logits_to_probs(x))

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

    cpu_device = jax.devices('cpu')[0]

    with jax.default_device(cpu_device):
        fit_nll_jax(Ks, sheet_keys, slots, booster_spec)


if __name__ == "__main__":
    from mk_ic import install
    install()
    main()
