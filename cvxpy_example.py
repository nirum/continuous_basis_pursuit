"""
example using cvxpy
author: Niru Maheswaranathan
07:40 PM Apr 15, 2014
"""

import numpy as np
import cvxpy as cp
import cvxopt

if __name__=="__main__":

    # Problem data.
    n = 10
    m = 50
    A = cvxopt.normal(n,m)
    b = cvxopt.normal(n)
    gamma = cp.Parameter(sign="positive", name='gamma')

    # Construct the problem.
    x = cp.Variable(m,name='x')
    objective = cp.Minimize(sum(cp.square(A*x - b)) + gamma*cp.norm(x, 1))
    constraints = [0 <= x]
    p = cp.Problem(objective, constraints)

    # Assign a value to gamma and find the optimal x.
    def get_x(gamma_value):
        gamma.value = gamma_value
        result = p.solve()
        return x.value

    gammas = np.logspace(-2, -1, num=2)

    # Serial computation.
    x_values = [get_x(value) for value in gammas]
