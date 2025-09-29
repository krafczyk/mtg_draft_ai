import sympy as sp
import numpy as np
import scipy


def get_cluster(mdl, i, x_series):
    labels = mdl.predict(x_series.to_numpy().reshape(-1,1))
    return x_series[labels == i]


def copies_probability(sheet_slot_probs):
    copies = []
    probs = []

    # probability generating polynomial
    z = sp.Symbol('z')
    G = sp.prod((1-p_i) + p_i*z for p_i in sheet_slot_probs)
    Gz = sp.expand(G)

    for n in range(0, 10):
        coeff = Gz.coeff(z**n)
        if coeff == 0:
            continue
        copies.append(n)
        probs.append(coeff)

    return copies, probs


def vec_to_dict(vec, sym_list):
    result = {}
    for i, sym in enumerate(sym_list):
        result[sym] = vec[i]
    return result


def dict_to_vec(d, sym_list):
    result = []
    for sym in sym_list:
        result.append(d[sym])
    return np.array(result)


def minimize_sp_loss(loss, x0=None):
    sym_list = list(loss.free_symbols)

    loss_fn = sp.lambdify(sym_list, loss, 'numpy')
    loss_lambda = lambda x: loss_fn(*x)

    grad_expr = list(sp.Matrix([loss]).jacobian(sym_list)[0, :])
    grad = sp.lambdify(sym_list, grad_expr, "numpy")
    def jac(*t):
        return np.asarray(grad(*t), dtype=np.float32)
    jac_lambda = lambda x: jac(*x)

    if x0 is None:
        x0 = np.random.random(size=(len(sym_list),))/len(sym_list)
    else:
        x0 = dict_to_vec(x0, sym_list)

    sol = scipy.optimize.minimize(loss_lambda, jac=jac_lambda, x0=x0, bounds=[(0.,1.)]*len(sym_list))

    # Get parameter error from hessian
    #sol_hess = sp.Matrix(grad_expr).jacobian(sym_list)
    #err = np.linalg.pinv(np.asarray(sp.lambdify(sym_list, sol_hess, "numpy")(*sol.x), dtype=np.float32))

    return sol, vec_to_dict(sol.x, sym_list)
