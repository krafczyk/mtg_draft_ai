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

    print(f"Fit results:")
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

# def fit_nll_sympy_jax(Ks, sheet_keys, slots, booster_spec):
#     # Validate booster_spec
#     slot_ids = set(slots.keys())
#     for slot in booster_spec:
#         if slot not in slot_ids:
#             raise ValueError(f"Booster spec contains slot '{slot}' not found in slots: {slot_ids}")

#     # Validate slots
#     for slot_name, slot_sheets in slots.items():
#         for sheet in slot_sheets:
#             if sheet not in sheet_keys:
#                 raise ValueError(f"Slot '{slot_name}' contains sheet '{sheet}' not found in sheets: {list(sheet_keys.keys())}")

#     # Validate Ks
#     if set(Ks.index.names) != set(sheet_keys):
#         raise ValueError(f"Ks index names {Ks.index.names} do not match sheets {sheet_keys}")

#     L = len(booster_spec)
#     S = len(sheet_keys)
#     N = len(Ks)

#     # Extract numpy arrays from Ks and validate their shapes
#     K_np = Ks.index.to_frame(index=False).to_numpy(dtype=np.int32)
#     N_np = Ks.to_numpy(dtype=np.float32)
#     assert K_np.shape == (N, S), f"Expected K shape {(N,S)}, got {K_np.shape}"
#     assert N_np.shape == (N,), f"Expected N shape {(N,)}, got {N_np.shape}"
#     ic(K_np.shape, N_np.shape)

#     # Create z variable for each sheet
#     z_vars = {sheet: sp.Symbol(f"z_{sheet}", real=True, positive=True) for sheet in sheet_keys}

#     # Create slot prob expressions
#     slot_probs = {}
#     prob_vector = []
#     for slot_key, slot_sheets in slots.items():
#         if len(slot_sheets) == 1:
#             slot_probs[slot_key] = {slot_sheets[0]: 1.}
#         else:
#             slot_probs[slot_key] = {}
#             for sheet in slot_sheets:
#                 p = sp.Symbol(f"p_{slot_key}_{sheet}", real=True, positive=True)
#                 prob_vector.append(p)
#                 slot_probs[slot_key][sheet] = p

#     ic(slot_probs, prob_vector)

#     # Build PGF
#     g = 1.

#     g = sp.prod([
#         sum([
#             slot_probs[slot_key].get(sheet,0.) * z_vars[sheet]
#             for sheet in slot_probs[slot_key]
#         ])
#         for slot_key in booster_spec
#     ])

#     gz = g.expand()

#     # Build negative log-likelihood
#     log_likelihood_terms = []
#     for k, n in zip(K_np, N_np):
#         z = sp.prod([z_vars[sheet]**ki for sheet, ki in zip(sheet_keys, k)])
#         prob_term = gz.coeff(z)
#         if prob_term == 0:
#             continue
#         log_likelihood_terms.append(n*sp.log(prob_term))
#     log_likelihood = sp.Add(*log_likelihood_terms)

#     nll_sp = -log_likelihood

#     assert len(nll_sp.free_symbols.difference(set(prob_vector))) == 0

#     nll_np_fn_1 = sp.lambdify(prob_vector, nll_sp, 'jax')

#     # pre compute slot logit extraction indices
#     logit_idx = {}
#     softmax_slice_starts = []
#     softmax_slice_sizes = []
#     num_pars = 0
#     for slot_key, slot_sheets in slots.items():
#         ic(slot_key, slot_sheets)
#         num_sheets = len(slot_sheets)
#         if num_sheets > 1:
#             softmax_slice_starts.append(num_pars)
#             softmax_slice_sizes.append(num_sheets)
#             for sheet in slot_sheets:
#                 logit_idx[(slot_key,sheet)] = num_pars
#                 num_pars += 1
#     print(f"There are {num_pars} free parameters")

#     # Prepare jax arrays
#     softmax_slice_starts = tuple(softmax_slice_starts)
#     softmax_slice_sizes = tuple(softmax_slice_sizes)

#     # Precompute scatter indices for prob placement, and place needed 1.s
#     ic(sheet_keys)
#     invert_slot_key = []
#     invert_sheet_key = []
#     slot_i = 0
#     for i, slot_key in enumerate(booster_spec):
#         ic(i, slot_key)
#         slot_sheets = slots[slot_key]
#         if len(slot_sheets) > 1:
#             for sheet_key in slot_sheets:
#                 j = sheet_keys.index(sheet_key)
#                 invert_slot_key.append(slot_key)
#                 invert_sheet_key.append(sheet_key)
#                 slot_i += 1

#     @jax.jit
#     def logits_to_probs(x):
#         return jnp.concatenate([
#             jnn.softmax(lax.dynamic_slice(x, (start,), (size,)))
#             for start, size in zip(softmax_slice_starts, softmax_slice_sizes)
#         ], axis=0)

#     # Build NLL function
#     @jax.jit
#     def nll_fn_logits(x):
#         # Input x is a flat packing of logits for each slot with multiple sheets        

#         # Compute/update x slices with softmax
#         return nll_np_fn_1(*logits_to_probs(x))

#     x0 = np.random.random(size=(num_pars,))

#     # value + gradient in one JIT-compiled call
#     val_and_grad = jax.jit(jax.value_and_grad(nll_fn_logits))

#     # SciPy expects (f, g) with NumPy types when jac=True
#     def scipy_obj(x_np):
#         x = jnp.asarray(x_np)
#         f, g = val_and_grad(x)
#         return float(f), np.asarray(g)

#     # Warm-up compile (optional, avoids first-call compile during minimize)
#     _ = scipy_obj(np.asarray(x0, dtype=np.float64))

#     res = minimize(scipy_obj,
#                    x0=np.asarray(x0, dtype=np.float64),
#                    method="L-BFGS-B",
#                    jac=True)

#     ic(res)

#     x_fit = res.x

#     p_fit = logits_to_probs(x_fit)

#     # 1) Compute hessian at solution
#     hess_fn = jax.jit(jax.hessian(nll_fn_logits))
#     H = np.asarray(hess_fn(x_fit))

#     # 2) Invert hessian to get covariance of logits
#     try:
#         cov_x = np.linalg.inv(H)
#     except np.linalg.LinAlgError:
#         cov_x = np.linalg.pinv(H) # common if using full softmax blocks

#     # 3) Jacobian of probs wrt logits at x_fit
#     Jp_fn = jax.jit(jax.jacrev(logits_to_probs))
#     Jp = np.asarray(Jp_fn(x_fit))

#     # 4) Delta method: Cov of probabilities and SEs
#     cov_p = Jp @ cov_x @ Jp.T

#     # guard tiny negative due to numerics
#     var_p = np.clip(np.diag(cov_p), 0.0, None)
#     se_p = np.sqrt(var_p)

#     z = 1.96

#     # map vector back to slot/sheet dict
#     result = {}
#     for i, (slot_key, sheet_key) in enumerate(zip(invert_slot_key, invert_sheet_key)):
#         if slot_key not in result:
#             result[slot_key] = {}
#         result[slot_key][sheet_key] = (
#             np.float32(p_fit[i]), np.float32(z*se_p[i]))

#     ic(result)


# Maybe v3??
def fit_v3_nll_sympy_jax(Ks, sheet_keys, slots, booster_spec):
    # Use SymPy to build NLL expression, then Jax for everything else
    # Doesn't converge

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
    logit_vector = []
    prob_logit_sub = {}
    for slot_key, slot_sheets in slots.items():
        if len(slot_sheets) == 1:
            slot_probs[slot_key] = {slot_sheets[0]: 1.}
        else:
            slot_probs[slot_key] = {}
            logit_group = []
            prob_group = []
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
                prob_logit_sub[p_i] = sp.exp(l_i)/denom

    print("================Vec=====")
    ic(logit_vector)
    for k, v in prob_logit_sub.items():
        print(f"{sp.sstr(k)} -> {sp.sstr(v)}")

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

    nll_p_sp = -log_likelihood

    ic(nll_p_sp)

    assert len(nll_p_sp.free_symbols.difference(set(prob_vector))) == 0

    nll_l_sp = nll_p_sp.subs(prob_logit_sub)

    ic(nll_l_sp)

    assert len(nll_l_sp.free_symbols.difference(set(logit_vector))) == 0

    nll_l_lambda = lambda x: np.float32(sp.lambdify(logit_vector, nll_l_sp, modules="numpy")(*x))

    nll_l_grad_sp = list(sp.Matrix([nll_l_sp]).jacobian(logit_vector)[0, :])

    nll_l_grad = sp.lambdify(logit_vector, nll_l_grad_sp, modules="numpy")
    def nll_l_jac(*t):
        return np.asarray(nll_l_grad(*t), dtype=np.float32)
    nll_l_jac_lambda = lambda x: nll_l_jac(*x)

    # Get parameter error from hessian
    #hess_expr = sp.Matrix(grad_expr).jacobian(sym_list)
    #hess_fn = sp.lambdify(sym_list, hess_expr, "numpy")
    #hess = np.asarray(hess_fn(*sol.x), dtype=np.float32)

    # build expressions for softmax
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
    #val_and_grad = jax.jit(jax.value_and_grad(nll_fn_logits))

    # SciPy expects (f, g) with NumPy types when jac=True
    #def scipy_obj(x_np):
    #    x = jnp.asarray(x_np)
    #    f, g = val_and_grad(x)
    #    return float(f), np.asarray(g)

    # Warm-up compile (optional, avoids first-call compile during minimize)
    #_ = scipy_obj(np.asarray(x0, dtype=np.float64))

    res = minimize(nll_l_lambda, jac=nll_l_jac_lambda, x0=x0)

    #res = minimize(scipy_obj,
    #               x0=np.asarray(x0, dtype=np.float64),
    #               method="L-BFGS-B",
    #               jac=True)

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
