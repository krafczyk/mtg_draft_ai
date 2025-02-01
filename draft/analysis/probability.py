import sympy
import numbers
import itertools


def build_sampling_probability(slot_probabilities: list[numbers.Real | sympy.Basic], num_successes: int = 1) -> sympy.Basic:
    """
    Given a list of probabilities to produce a desired outcome across different slots,
    Produce a full probability a given number of successes can occur.
    """

    if num_successes > len(slot_probabilities):    
        return sympy.core.Float(0)

    slot_prob_Is = list(range(len(slot_probabilities)))

    P = []
    for I in itertools.combinations(slot_prob_Is, num_successes):
        p = []
        for i in I:
            p.append(slot_probabilities[i])
        for j in slot_prob_Is:
            if j in I:
                continue
            p.append(sympy.core.Float(1)-slot_probabilities[j])
        p = sympy.Mul(*p)
        P.append(p)

    return sympy.Add(*P)
