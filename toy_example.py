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
    Fp = cvxopt.matrix(F)
    dFp = cvxopt.matrix(dF[0])
    yp = cvxopt.matrix(y)
    gamma = cp.Parameter(sign="positive", name='gamma')
    gamma.value = penalty

    x = cp.Variable(F.shape[1],name='x')
    d = cp.Variable(F.shape[1],name='d')
    objective = cp.Minimize(sum(cp.square(yp - Fp*x - dFp*d)) + gamma*cp.norm(x, 1))
    constraints = [0 <= x, cp.abs(d) <= 0.5*Delta*x]
    p = cp.Problem(objective, constraints)

    # solve
    result = p.solve()

    # reconstruct
    yhat = F.dot(np.array(x.value)) + dF[0].dot(np.array(d.value))

    return np.array(x.value), yhat, np.array(d.value), p.value

def cbp_polar(y, F, Fprev, Fnext, Delta, penalty=0.1, order=1):
    """
    CBP with polar interpolation
    """

    # compute r and theta
    a = 0.5 * np.linalg.norm(Fnext-Fprev, axis=0)[int(F.shape[1]/2)]
    b = np.linalg.norm(F-Fprev, axis=0)[int(F.shape[1]/2)]
    theta = np.pi - 2 * np.arcsin(a/b)      # radians
    r = a / np.sin(theta)

    # build the polar transformation matrix
    P = np.array([[1,r*np.cos(theta),-r*np.sin(theta)], [1,r,0], [1,r*np.cos(theta),r*np.sin(theta)]])

    # get C, U, and V
    pol = np.linalg.inv(P).dot(np.vstack((Fprev.ravel(),F.ravel(),Fnext.ravel())))
    C = pol[0,:].reshape(F.shape)
    U = pol[1,:].reshape(F.shape)
    V = pol[2,:].reshape(F.shape)

    ## construct the problem

    # discretized matrices
    Cp = cvxopt.matrix(C)
    Up = cvxopt.matrix(U)
    Vp = cvxopt.matrix(V)
    yp = cvxopt.matrix(y)

    # sparsity penalty
    gamma = cp.Parameter(sign="positive", name='gamma')
    gamma.value = penalty

    # variables
    dx = cp.Variable(F.shape[1],name='x')
    dy = cp.Variable(F.shape[1],name='y')
    dz = cp.Variable(F.shape[1],name='z')

    # objective and constraints
    objective = cp.Minimize(sum(cp.square(yp - Cp*dx - Up*dy - Vp*dz)) + gamma*cp.norm(dx, 1))
    #constraints = [0 <= x, cp.sqrt(cp.square(y)+cp.square(z)) <= r*x, r*np.cos(theta)*x <= y, y <= r*x]
    sqcon = [cp.norm(cp.vstack(yi,zi),2) <= xi*r for xi, yi, zi in zip(dx,dy,dz)]
    constraints = [0 <= dx, dy <= r*dx, r*np.cos(theta)*dx <= dy]
    constraints.extend(sqcon)
    p = cp.Problem(objective, constraints)

    # solve
    result = p.solve()

    # reconstruct
    yhat = C.dot(np.array(dx.value)) + U.dot(np.array(dy.value)) + V.dot(np.array(dz.value))

    return np.array(dx.value), yhat, p.value

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
    t = np.linspace(-25,25,m)
    f = basisfun(t, 0, sigma)

    # pick a time shift
    tau = -3

    # generate the signal
    eta = 0.05
    y = basisfun(t,tau,sigma) + eta * np.random.randn(t.size)

    # number of copies
    N = 4
    spacings = np.linspace(t.min()+10,t.max()-10,N)
    delta = np.mean(np.diff(spacings))

    # build the dictionary
    F = np.vstack([basisfun(t,d,sigma) for d in spacings]).T

    # half-spacings for polar interpolation
    Fprev = np.vstack([basisfun(t,d-delta/2.0,sigma) for d in spacings]).T
    Fnext = np.vstack([basisfun(t,d+delta/2.0,sigma) for d in spacings]).T

    # run basis pursuit
    xhat_bp, yhat_bp = basispursuit(y, F, penalty=0.01)

    # run 1st order taylor approximation
    xhat_taylor, yhat_taylor, dhat_taylor, objval_taylor = cbp_taylor(y, F, delta, penalty=0.1)

    # run polar interpolation
    xhat_polar, yhat_polar, objval_polar = cbp_polar(y, F, Fprev, Fnext, delta, penalty=0.1)

    plt.figure()
    plt.plot(t,y,'k-', t, yhat_bp, 'b--', t, yhat_taylor, 'r--', t, yhat_polar, 'g--')
    plt.show()
