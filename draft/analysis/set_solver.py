from draft.datasets.seventeenlands import get_pack_data_summary_dense, get_card_df
from draft.datasets.ops import attach_card_metadata
from draft.booster.basic import PrintSheet, BoosterSlot
from draft.booster.booster import BoosterModel
from draft.utils import sp_evaluate
import pandas as pd
import numpy as np
import sympy as sp
from scipy.optimize import minimize
from scipy.special import softmax as np_softmax
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

def fit_v1_nll_jax(Ks, sheet_keys, slots, booster_spec):
    # v1 - NLL with only Jax 
    # CPU only, very memory intensive. Wants to allocate a 14^6 size array to build full PDF
    # OOM with GTX1080 GPU.

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

    # pre compute slot logit extraction indices
    logit_idx = {}
    softmax_slice_starts = []
    softmax_slice_sizes = []
    num_pars = 0
    for slot_key, slot_sheets in slots.items():
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

    # Precompute initial slot prob structure
    init_slot_probs = np.zeros((L, S), dtype=np.float32)

    # Precompute scatter indices for prob placement, and place needed 1.s
    out_rows, out_cols = [], []
    invert_slot_key = []
    invert_sheet_key = []
    slot_i = 0
    for i, slot_key in enumerate(booster_spec):
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

    print(f"Fit status:")
    print(res)

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

    return result

def fit_v2_nll_sympy_jax(Ks, sheet_keys, slots, booster_spec):
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
    invert_slot_key = []
    invert_sheet_key = []
    slot_i = 0
    for i, slot_key in enumerate(booster_spec):
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
    _ = scipy_obj(np.asarray(x0, dtype=np.float32))

    res = minimize(scipy_obj,
                   x0=np.asarray(x0, dtype=np.float32),
                   method="L-BFGS-B",
                   jac=True)

    print(f"Fit status:")
    print(res)

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

    return result


def fit_v3_nll_sympy_jax(Ks, sheet_keys, slots, booster_spec):
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

    # Create z variable for each sheet
    z_vars = {sheet: sp.Symbol(f"z_{sheet}", real=True, positive=True) for sheet in sheet_keys}

    # Create slot prob expressions
    slot_probs = {}
    prob_vector = []
    logit_vector = []

    for slot_key, slot_sheets in slots.items():
        if len(slot_sheets) == 1:
            slot_probs[slot_key] = {slot_sheets[0]: 1.}
        else:
            slot_probs[slot_key] = {}
            group_p = []
            group_l = []
            for sheet in slot_sheets:
                p = sp.Symbol(f"p_{slot_key}_{sheet}", real=True, positive=True)
                group_p.append(p)
                group_l.append(sp.Symbol(f"l_{slot_key}_{sheet}", real=True))
                prob_vector.append(p)
                slot_probs[slot_key][sheet] = p

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

    nll_np_fn_1 = sp.lambdify(prob_vector, nll_sp, 'jax', cse=True)

    # pre compute slot logit extraction indices
    logit_idx = {}
    softmax_slice_starts = []
    softmax_slice_sizes = []
    num_pars = 0
    for slot_key, slot_sheets in slots.items():
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
    invert_slot_key = []
    invert_sheet_key = []
    slot_i = 0
    for i, slot_key in enumerate(booster_spec):
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

    print(f"Fit status:")
    print(res)

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

    return result

def fit_v4_nll_sympy_jax(Ks, sheet_keys, slots, booster_spec):
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

    # Create z variable for each sheet
    z_vars = {sheet: sp.Symbol(f"z_{sheet}", real=True, positive=True) for sheet in sheet_keys}

    # Create slot prob expressions
    slot_probs = {}
    prob_vector = []
    logit_vector = []
    prob_logit_vector = []
    softmax_slice_starts = []
    softmax_slice_sizes = []
    num_pars = 0
    for slot_key, slot_sheets in slots.items():
        if len(slot_sheets) == 1:
            slot_probs[slot_key] = {slot_sheets[0]: 1}
        else:
            slot_probs[slot_key] = {}
            logit_group = []
            prob_group = []
            softmax_slice_starts.append(num_pars)
            softmax_slice_sizes.append(len(slot_sheets))
            num_pars += len(slot_sheets)
            for sheet in slot_sheets:
                p = sp.Symbol(f"p_{slot_key}_{sheet}", real=True, positive=True)
                l = sp.Symbol(f"l_{slot_key}_{sheet}", real=True)
                prob_vector.append(p)
                logit_vector.append(l)
                prob_group.append(p)
                logit_group.append(l)
                slot_probs[slot_key][sheet] = p
            denom = sp.Add(*[sp.exp(l_i) for l_i in logit_group])
            for p_i, l_i in zip(prob_group, logit_group):
                prob_logit_vector.append(sp.exp(l_i)/denom)

    softmax_slice_starts = tuple(softmax_slice_starts)
    softmax_slice_sizes = tuple(softmax_slice_sizes)

    # Precompute scatter indices for prob placement, and place needed 1.s
    invert_slot_key = []
    invert_sheet_key = []
    slot_i = 0
    for i, slot_key in enumerate(booster_spec):
        slot_sheets = slots[slot_key]
        if len(slot_sheets) > 1:
            for sheet_key in slot_sheets:
                j = sheet_keys.index(sheet_key)
                invert_slot_key.append(slot_key)
                invert_sheet_key.append(sheet_key)
                slot_i += 1

    # Build PGF
    g = sp.prod([
        sum([
            slot_probs[slot_key].get(sheet,0.) * z_vars[sheet]
            for sheet in slot_probs[slot_key]
        ])
        for slot_key in booster_spec
    ])

    gz = g.expand()

    P_terms = []
    for k in K_np:
        z = sp.prod([z_vars[sheet]**ki for sheet, ki in zip(sheet_keys, k)])
        P_terms.append(gz.coeff(z))

    # Build negative log-likelihood
    log_likelihood_terms = []
    for P, n in zip(P_terms, N_np):
        log_likelihood_terms.append(n*sp.log(P))
    log_likelihood = sp.Add(*log_likelihood_terms)

    nll_p_sp = -log_likelihood
    assert len(nll_p_sp.free_symbols.difference(set(prob_vector))) == 0

    J_pl = sp.Matrix(prob_logit_vector).jacobian(logit_vector).subs({pl: p for pl, p in zip(prob_logit_vector, prob_vector)})

    J_nll_p = sp.Matrix([nll_p_sp]).jacobian(prob_vector)

    jac_p = J_nll_p @ J_pl

    nll_p_jax = sp.lambdify(prob_vector, nll_p_sp, cse=True, modules="jax")
    jac_p_jax = sp.lambdify(prob_vector, jac_p, cse=True, modules="jax")
    hess_p = jac_p.jacobian(prob_vector)
    hess_p_jax = sp.lambdify(prob_vector, hess_p, cse=True, modules="jax")

    # Define jax methods
    @jax.jit
    def logits_to_probs(x):
        return jnp.concatenate([
            jnn.softmax(lax.dynamic_slice(x, (start,), (size,)))
            for start, size in zip(softmax_slice_starts, softmax_slice_sizes)
        ], axis=0)

    @jax.jit
    def nll_l_val_grad_fn(x):
        p = logits_to_probs(x)
        #return nll_p_jax(*logits_to_probs(x)), jac_nll_l_fn(x)
        return nll_p_jax(*p), jac_p_jax(*p)

    x0 = np.random.random(size=(num_pars,))

    def scipy_obj(x_np):
        x = jnp.asarray(x_np)
        f, g = nll_l_val_grad_fn(x)
        return np.float64(f), np.asarray(g)

    _ = scipy_obj(np.asarray(x0, dtype=np.float64))

    res = minimize(scipy_obj, x0=np.asarray(x0, dtype=np.float64), method="L-BFGS-B", jac=True)
    p_fit = logits_to_probs(res.x)

    H = np.array(hess_p_jax(*logits_to_probs(res.x)))

    # 2) Invert hessian to get covariance of logits
    try:
        cov_p = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_p = np.linalg.pinv(H) # common if using full softmax blocks

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

    return result

def fit_v5_nll_sympy_numpy(Ks, sheet_keys, slots, booster_spec, dtype=np.float32, tol=None):
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
    N_np = Ks.to_numpy(dtype=dtype)
    assert K_np.shape == (N, S), f"Expected K shape {(N,S)}, got {K_np.shape}"
    assert N_np.shape == (N,), f"Expected N shape {(N,)}, got {N_np.shape}"

    # Create z variable for each sheet
    z_vars = {sheet: sp.Symbol(f"z_{sheet}", real=True, positive=True) for sheet in sheet_keys}

    # Create slot prob expressions
    slot_probs = {}
    prob_vector = []
    logit_vector = []
    prob_logit_vector = []
    softmax_slice_starts = []
    softmax_slice_sizes = []
    num_pars = 0
    for slot_key, slot_sheets in slots.items():
        if len(slot_sheets) == 1:
            slot_probs[slot_key] = {slot_sheets[0]: 1}
        else:
            slot_probs[slot_key] = {}
            logit_group = []
            prob_group = []
            softmax_slice_starts.append(num_pars)
            softmax_slice_sizes.append(len(slot_sheets))
            num_pars += len(slot_sheets)
            for sheet in slot_sheets:
                p = sp.Symbol(f"p_{slot_key}_{sheet}", real=True, positive=True)
                l = sp.Symbol(f"l_{slot_key}_{sheet}", real=True)
                prob_vector.append(p)
                logit_vector.append(l)
                prob_group.append(p)
                logit_group.append(l)
                slot_probs[slot_key][sheet] = p
            denom = sp.Add(*[sp.exp(l_i) for l_i in logit_group])
            for p_i, l_i in zip(prob_group, logit_group):
                prob_logit_vector.append(sp.exp(l_i)/denom)

    softmax_slice_starts = tuple(softmax_slice_starts)
    softmax_slice_sizes = tuple(softmax_slice_sizes)

    # Precompute scatter indices for prob placement, and place needed 1.s
    invert_slot_key = []
    invert_sheet_key = []
    slot_i = 0
    for i, slot_key in enumerate(booster_spec):
        slot_sheets = slots[slot_key]
        if len(slot_sheets) > 1:
            for sheet_key in slot_sheets:
                j = sheet_keys.index(sheet_key)
                invert_slot_key.append(slot_key)
                invert_sheet_key.append(sheet_key)
                slot_i += 1

    # Build PGF
    g = sp.prod([
        sum([
            slot_probs[slot_key].get(sheet,0.) * z_vars[sheet]
            for sheet in slot_probs[slot_key]
        ])
        for slot_key in booster_spec
    ])

    gz = g.expand()

    P_terms = []
    for k in K_np:
        z = sp.prod([z_vars[sheet]**ki for sheet, ki in zip(sheet_keys, k)])
        P_terms.append(gz.coeff(z))

    # Build negative log-likelihood
    log_likelihood_terms = []
    for P, n in zip(P_terms, N_np):
        log_likelihood_terms.append(n*sp.log(P))
    log_likelihood = sp.Add(*log_likelihood_terms)

    nll_p_sp = -log_likelihood
    assert len(nll_p_sp.free_symbols.difference(set(prob_vector))) == 0

    J_pl = sp.Matrix(prob_logit_vector).jacobian(logit_vector).subs({pl: p for pl, p in zip(prob_logit_vector, prob_vector)})

    J_nll_p = sp.Matrix([nll_p_sp]).jacobian(prob_vector)

    jac_p = J_nll_p @ J_pl

    nll_p_np = sp.lambdify(prob_vector, nll_p_sp, cse=True, modules="numpy")
    jac_p_np = sp.lambdify(prob_vector, jac_p, cse=True, modules="numpy")
    hess_p = jac_p.jacobian(prob_vector)
    hess_p_np = sp.lambdify(prob_vector, hess_p, cse=True, modules="numpy")

    # Define numpy methods
    def logits_to_probs(x):
        return np.concatenate([
            np_softmax(x[start:start+size])
            for start, size in zip(softmax_slice_starts, softmax_slice_sizes)
        ], axis=0)

    def nll_l_val_grad_fn(x):
        p = logits_to_probs(x)
        return nll_p_np(*p), jac_p_np(*p)

    x0 = np.asarray(np.random.random(size=(num_pars,)), dtype=dtype)

    def scipy_obj(x_np):
        f, g = nll_l_val_grad_fn(x_np)
        return dtype(f), np.asarray(g)

    _ = scipy_obj(x0)

    res = minimize(scipy_obj, x0=x0, method="L-BFGS-B", jac=True)
    print(f"Fit status:")
    print(res)
    p_fit = logits_to_probs(res.x)

    H = np.array(hess_p_np(*logits_to_probs(res.x)))

    # 2) Invert hessian to get covariance of logits
    try:
        cov_p = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_p = np.linalg.pinv(H) # common if using full softmax blocks

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
            p_fit[i], z*se_p[i])

    return result

def fit_v6_nll_sympy_numpy(Ks, sheet_keys, slots, booster_spec, known_p=None, dtype=np.float32, tol=None):
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

    if known_p is None:
        known_p = {}

    L = len(booster_spec)
    S = len(sheet_keys)
    N = len(Ks)

    # Extract numpy arrays from Ks and validate their shapes
    K_np = Ks.index.to_frame(index=False).to_numpy(dtype=np.int32)
    N_np = Ks.to_numpy(dtype=dtype)
    assert K_np.shape == (N, S), f"Expected K shape {(N,S)}, got {K_np.shape}"
    assert N_np.shape == (N,), f"Expected N shape {(N,)}, got {N_np.shape}"

    # Create z variable for each sheet
    z_vars = {sheet: sp.Symbol(f"z_{sheet}", real=True, positive=True) for sheet in sheet_keys}

    # Create slot prob expressions
    slot_probs = {}
    x_vector = []
    e_vector = []
    l_vector = []
    xl_vector = []
    softmax_slice_starts = []
    softmax_slice_sizes = []
    num_pars = 0
    for slot_key, slot_sheets in slots.items():
        if len(slot_sheets) == 1:
            slot_probs[slot_key] = {slot_sheets[0]: (1, 0)}
            continue

        slot_probs[slot_key] = {}
        # We already know these probabilities
        if slot_key in known_p and len(slot_sheets) == len(known_p[slot_key]):
            # Check that known_p sums to 1
            assert abs(sum(known_p[slot_key].values()) - 1) < 1e-8, f"Known probabilities for slot '{slot_key}' do not sum to 1"
            for sheet in slot_sheets:
                p = known_p[slot_key][sheet]
                slot_probs[slot_key][sheet] = (p,0)
            continue
        elif slot_key in known_p and len(slot_sheets) == len(known_p[slot_key])+1:
            # Find/Solve remaining p
            rem_sheet = set(slot_sheets).difference(set(known_p[slot_key].keys()))
            assert len(rem_sheet) == 1, f"Slot '{slot_key}' has more than one unknown probability"
            known_p[slot_key][rem_sheet.pop()] = 1 - sum(known_p[slot_key].values())
            for sheet in slot_sheets:
                p = known_p[slot_key][sheet]
                slot_probs[slot_key][sheet] = (p,0)
            continue

        l_group = []
        x_group = []

        known_probs = []
        known_sheets = []
        if slot_key in known_p:
            known_probs = list(known_p[slot_key].values())
            known_sheets = list(known_p[slot_key].keys())
        logit_norm = 1 - sum(known_probs)

        num_pars_sheet = 0
        for sheet in slot_sheets:
            if sheet in known_sheets:
                slot_probs[slot_key][sheet] = (known_p[slot_key][sheet], 0)
                continue

            x = sp.Symbol(f"x_{slot_key}_{sheet}", real=True, positive=True)
            e = sp.Symbol(f"e_{slot_key}_{sheet}", real=True, positive=True)
            l = sp.Symbol(f"l_{slot_key}_{sheet}", real=True)
            x_group.append(x)
            l_group.append(l)
            x_vector.append(x)
            l_vector.append(l)
            e_vector.append(e)
            slot_probs[slot_key][sheet] = (x*logit_norm, e*logit_norm)
            num_pars_sheet += 1
        denom = sp.Add(*[sp.exp(l_i) for l_i in l_group])
        for l_i in l_group:
            xl_vector.append(sp.exp(l_i)/denom)
        softmax_slice_starts.append(num_pars)
        softmax_slice_sizes.append(num_pars_sheet)
        num_pars += num_pars_sheet

    softmax_slice_starts = tuple(softmax_slice_starts)
    softmax_slice_sizes = tuple(softmax_slice_sizes)

    # Build PGF
    g = sp.prod([
        sum([
            slot_probs[slot_key].get(sheet,0.)[0] * z_vars[sheet]
            for sheet in slot_probs[slot_key]
        ])
        for slot_key in booster_spec
    ])

    gz = g.expand()

    P_terms = []
    for k in K_np:
        z = sp.prod([z_vars[sheet]**ki for sheet, ki in zip(sheet_keys, k)])
        P_terms.append(gz.coeff(z))

    # Build negative log-likelihood
    log_likelihood_terms = []
    for P, n in zip(P_terms, N_np):
        log_likelihood_terms.append(n*sp.log(P))
    log_likelihood = sp.Add(*log_likelihood_terms)

    nll_sp = -log_likelihood
    assert len(nll_sp.free_symbols.difference(set(x_vector))) == 0
    # Apply simplification works well here
    expanded = sp.expand_log(nll_sp)
    logs = sorted(expanded.atoms(), key=str)
    nll_sp = sp.collect(expanded, logs)

    # Build p(l) transform jacobian
    J_xl = sp.Matrix(xl_vector).jacobian(l_vector)
    # Use dummy to help SymPy keep things straight

    J_xl = J_xl.subs({xl: x for xl, x in zip(xl_vector, x_vector)})

    unexpected = J_xl.free_symbols.difference(set(x_vector))
    assert len(unexpected) == 0, f"J_xl has unexpected free symbols: {unexpected}"

    J_nll = sp.Matrix([nll_sp]).jacobian(x_vector)

    jac_x = J_nll @ J_xl

    nll_np = sp.lambdify(x_vector, nll_sp, cse=True, modules="numpy")
    jac_np = sp.lambdify(x_vector, jac_x, cse=True, modules="numpy")
    hess_x = jac_x.jacobian(x_vector)
    hess_x_np = sp.lambdify(x_vector, hess_x, cse=True, modules="numpy")

    # Define numpy methods
    def logits_to_x(l):
        return np.concatenate([
            np_softmax(l[start:start+size])
            for start, size in zip(softmax_slice_starts, softmax_slice_sizes)
        ], axis=0)

    def nll_l_val_grad_fn(l):
        x = logits_to_x(l)
        return nll_np(*x), jac_np(*x)

    x0 = np.asarray(np.random.random(size=(num_pars,)), dtype=dtype)

    def scipy_obj(x_np):
        f, g = nll_l_val_grad_fn(x_np)
        return dtype(f), np.asarray(g)

    _ = scipy_obj(x0)

    res = minimize(scipy_obj, x0=x0, method="L-BFGS-B", jac=True)
    print(f"Fit status:")
    print(res)
    x_fit = logits_to_x(res.x)

    H = np.array(hess_x_np(*logits_to_x(res.x)))

    # 2) Invert hessian to get covariance of logits
    try:
        cov_x = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov_x = np.linalg.pinv(H) # common if using full softmax blocks

    # guard tiny negative due to numerics
    var_x = np.clip(np.diag(cov_x), 0.0, None)
    se_x = np.sqrt(var_x)

    z = 1.96

    answer_subs = {x: xf for x, xf in zip(x_vector, x_fit)} \
        | { e: z*se for e, se in zip(e_vector, se_x) }

    return sp_evaluate(slot_probs, subs=answer_subs)
