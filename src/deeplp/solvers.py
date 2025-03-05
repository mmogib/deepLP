from scipy.optimize import linprog


def solve_lp(D, A, b):
    res = linprog(D, A_ub=A, b_ub=b, bounds=(None, None), method="highs")
    assert res.success, "Solver found no solution"
    return res.x, res.fun
