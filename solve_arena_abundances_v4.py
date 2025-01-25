from sympy import symbols, Eq, solve
from pprint import pprint
import numpy as np

# Forge (current)
#C = 7.3923 - 6
#U = 4.2150 - 3
#R = 1.1820
#M = 0.1954
#SPG = 0.0154

# Arena
C = 7.7687 - 6
U = 3.9223 - 3
R = 1.1086
M = 0.1849
SPG = 0.0156

# Only slot 7 can produce spg. This forces the value for P_spg and P_c1
P_spg = SPG
P_c1 = 1-P_spg # P_c1 + P_spg = 1

results = {
    'spg': P_spg,
    'c1': P_c1,
}

# Adjusted remaining abundance
C2 = C-P_c1

# Only wildcard slots can produce commons at this point, so we can solve for P_c2 since C2 = 2*P_c2.

P_c2 = C2/2

results['c2'] = P_c2

# Only wildcard slots can produce uncommons after removing guaranteed slots. so we have U = 2*P_u2

P_u2 = U/2

results['u2'] = P_u2

M_p = np.array([
    [ 1, 0, 2, 0],
    [ 0, 1, 0, 2],
    [ 1, 1, 0, 0],
    [ 0, 0, 1, 1],
])

print(f"Determinant: {np.linalg.det(M_p)}") # Not solvable!

# Assume P_r1 is as written: 0.857, then P_m1 = 0.143 from (P_r1+P_m1 = 1)
# Then, We have:

P_r1 = 0.857
P_m1 = 0.143

results['r1'] = 0.857
results['m1'] = 0.143

P_r2 = (R-P_r1)/2
P_m2 = 1-P_c2-P_u2-P_r2

results['r2'] = P_r2
results['m2'] = P_m2

# Finally, we then have
# R               = P_r1 +        + 2*P_r2;
# M               =          P_m1 +        + 2*P_m2;
# 1               = P_r1 +   P_m1;
# 1 - P_c2 - P_u2 =                   P_r2 +   P_m2;

# Matrix is:
# 1 0 2 0
# 0 1 0 2
# 1 1 0 0
# 0 0 1 1

pprint(results)
assert abs(C+U+R+M+SPG-4) < 1e-3
