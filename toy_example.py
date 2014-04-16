"""
Toy 2D example for continuous basis pursuit
author: Niru Maheswaranathan, Ben Naecker, Ben Poole
06:42 PM Apr 15, 2014
"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import cvxopt
from sklearn.linear_model import Lasso

def basispursuit(y, F, penalty=0.1):
    """
    solves basic (vanilla) basis pursuit using scikit-learn
    """

    clf = Lasso(alpha=penalty, fit_intercept=False)
    clf.fit(F, y)
    xhat = clf.coef_

    # reconstruct
    yhat = F.dot(xhat)

    return xhat, yhat

def cbp_taylor(y, F, Delta, penalty=0.1, order=1):
    """
    1st order taylor approximation
    """

    # generate derivative matrices
    dF = list()
    current = F
    for i in range(order):
        dF.append( deriv(current,t) )
        current = dF[-1]

    # Construct the problem.
    Fcp = cvxopt.matrix(F)
    dFcp = cvxopt.matrix(dF[0])
    ycp = cvxopt.matrix(y)
    gamma = cp.Parameter(sign="positive", name='gamma')
    gamma.value = penalty

    x = cp.Variable(F.shape[1],name='x')
    d = cp.Variable(F.shape[1],name='d')
    objective = cp.Minimize(sum(cp.square(ycp - Fcp*x - dFcp*d)) + gamma*cp.norm(x, 1))
    constraints = [0 <= x, cp.abs(d) <= 0.5*Delta*x]
    p = cp.Problem(objective, constraints)

    # solve
    result = p.solve()

    # reconstruct
    yhat = F.dot(np.array(x.value)) + dF[0].dot(np.array(d.value))

    return np.array(x.value), yhat, np.array(d.value), p.value

def basisfun(t,tau,sigma):
    """
    return the basis function shifted by some amount tau
    """

    return np.exp(- (t-tau) **2 / sigma**2 )

def deriv(X,t):
    """
    estimates the derivative of columns of a matrix X(t)
    """
    dxdt = np.diff(X,axis=0) / np.mean(np.diff(t))
    return np.vstack(( dxdt, np.zeros((1, X.shape[1])) ))

if __name__ == "__main__":

    # generate 2D interpolation

    # pick a basis function
    m = 25
    sigma = 5
    t = np.linspace(-10,10,m)
    f = basisfun(t, 0, sigma)

    # pick a time shift
    tau = -3

    # generate the signal
    eta = 0.05
    y = basisfun(t,tau,sigma) + eta * np.random.randn(t.size)

    # number of copies
    N = 5
    delta = np.linspace(t.min(),t.max(),N)

    # build the dictionary
    F = np.vstack([basisfun(t,d,sigma) for d in delta]).T

    # run basis pursuit
    xhat_bp, yhat_bp = basispursuit(y, F, penalty=0.1)

    # run 1st order taylor approximation
    xhat_taylor, yhat_taylor, dhat_taylor, objval = cbp_taylor(y, F, np.mean(np.diff(delta)), penalty=0.1)
    #dF = cbp_taylor(y, F, penalty=0.1)
    #print dF

    plt.figure()
    plt.plot(t,y,'k-', t, yhat_bp, 'b--', t, yhat_taylor, 'r--')
    plt.show()
