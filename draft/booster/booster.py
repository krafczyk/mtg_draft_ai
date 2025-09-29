import numpy as np
import sympy as sp
import pandas as pd
import scipy

from draft.booster.basic import PrintSheet, BoosterSlot
from draft.typing import RealLike
import draft.analysis.utils as utils


class BoosterModel:
    slots: list[BoosterSlot]
    sheets: dict[str,PrintSheet]

    def __init__(self, slots: list[BoosterSlot]):
        self.slots = slots
        # Check that all sheets are mutually exclusive
        unique_sheets = self.get_unique_sheets()
        self.sheets = {}
        for i in range(len(unique_sheets)):
            self.sheets[unique_sheets[i].name] = unique_sheets[i]
            for j in range(i + 1, len(unique_sheets)):
                sheet_i = unique_sheets[i]
                sheet_j = unique_sheets[j]
                names_i = set(sheet_i.cards.keys())
                names_j = set(sheet_j.cards.keys())
                intersection = names_i.intersection(names_j)
                if intersection:
                    raise ValueError(f"Sheets {sheet_i.name} and {sheet_j.name} share the following cards: {intersection}")

    def get_card_name_list(self) -> list[str]:
        unique_sheets = self.get_unique_sheets()
        all_cards = []
        for sheet in unique_sheets:
            all_cards.extend(list(map(lambda c: c.name, sheet.cards.keys())))

        all_cards.sort()
        return all_cards

    def get_unique_sheets(self) -> list[PrintSheet]:
        seen = set()
        unique_sheets = []
        for slot in self.slots:
            for sheet_spec in slot.sheets:
                if sheet_spec.sheet.name not in seen:
                    seen.add(sheet_spec.sheet.name)
                    unique_sheets.append(sheet_spec.sheet)
        return unique_sheets

    def set_slot_probs(self, prob_dict: dict[str|sp.Symbol,RealLike]|None=None):
        for slot in self.slots:
            slot.set_slot_probs(prob_dict=prob_dict)

    def model_prob_summary(self):
        unique_sheets = self.get_unique_sheets()

        probs = {}

        for sheet in unique_sheets:
            probs[sheet.name] = []
            for slot in self.slots:
                slot_prob = 0.
                for sheet_spec in slot.sheets:
                    if sheet_spec.sheet.name == sheet.name:
                        slot_prob = sheet_spec.prob
                        break
                probs[sheet.name].append(slot_prob)

        return probs
        

    def fit(self, df: pd.DataFrame, x0=None):
        all_eqs = []
        loss_w1 = 2.

        loss = 0.

        mdl_probs = self.model_prob_summary()

        for sheet_name, sheet in self.sheets.items():
            sheet_slot_probs = mdl_probs[sheet_name]
            n, p = utils.copies_probability(sheet_slot_probs)
            for ni, pi in zip(n, p):
                f = (df.loc[:,sheet.card_names].sum(axis=1) == ni).sum()/len(df)

                loss += (f - pi)**2

        for slot in self.slots:
            p_sum = sum(spec.prob for spec in slot.sheets)
            loss += 2*(p_sum - 1)**2

        return utils.minimize_sp_loss(loss, x0=x0)

    def fit_1st_order(self, df: pd.DataFrame) -> dict[str,np.float32]:
        # Build an equation for each unique print sheet
        unique_sheets = self.get_unique_sheets()

        sheet_ev: dict[str, np.float32] = {}
        for sheet in unique_sheets:
            # Measure the number of times this sheet was sampled
            ev = np.float32(0.)
            for subsheet_weight, cards in sheet.sub_sheets().items():
                # Pick only those packs that have a single card from this sub-sheet
                ev += df.loc[:,cards].sum(axis=1).mean()*subsheet_weight
            sheet_ev[sheet.name] = ev

        print(sheet_ev)

        equations = {}

        # Build equations
        for sheet_name in sheet_ev:
            total_p = 0.
            sheet_probs = []
            for slot in self.slots:
                # Check if this slot contains this sheet
                for sheet_spec in slot.sheets:
                    if sheet_spec.sheet.name == sheet_name:
                        total_p += sheet_spec.prob

            if hasattr(total_p, 'free_symbols'):
                if len(total_p.free_symbols) != 0:
                    equations[sheet_name] = total_p - sheet_ev[sheet_name]

                    print(f"Equation for sheet {sheet_name}: {total_p} = {sheet_ev[sheet_name]}")
                    continue
            print(f"Warning: sheet {sheet_name} has constant probability {total_p}, skipping")

        # Solve equations
        all_syms = set().union(*(eq.free_symbols for eq in equations.values()))
        # Exact solution likely not possible due to noise. Use numerical solver instead.
        sym_list = list(all_syms)

        loss_sp = sum(eq**2 for eq in equations.values())
        loss_fn = sp.lambdify(sym_list, loss_sp, 'numpy')
        loss_lambda = lambda x: loss_fn(*x)

        grad_expr = list(sp.Matrix([sum(eq**2 for eq in equations.values())]).jacobian(sym_list)[0, :])
        grad = sp.lambdify(sym_list, grad_expr, "numpy")
        def jac(*t):
            return np.asarray(grad(*t), dtype=np.float32)
        jac_lambda = lambda x: jac(*x)

        x0 = np.array([1/len(all_syms)]*len(all_syms))

        sol = scipy.optimize.minimize(loss_lambda, jac=jac_lambda, x0=x0, bounds=[(0.,1.)]*len(sym_list))

        # Get parameter error from hessian
        sol_hess = sp.Matrix(grad_expr).jacobian(sym_list)
        err = np.linalg.pinv(np.asarray(sp.lambdify(sym_list, sol_hess, "numpy")(*sol.x), dtype=np.float32))

        return sol.x, err
