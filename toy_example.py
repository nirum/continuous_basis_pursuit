"""
Toy 2D example for continuous basis pursuit
author: Niru Maheswaranathan, Ben Naecker, Ben Poole
06:42 PM Apr 15, 2014
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def basispursuit(y, F, penalty=0.1):

    clf = Lasso(alpha=penalty, fit_intercept=False)
    clf.fit(F, y)
    xhat = clf.coef_

    # reconstruct
    yhat = F.dot(xhat)

    return xhat, yhat

def basisfun(t,tau,sigma):
    """
    return the basis function shifted by some amount tau
    """

    return np.exp(- (t-tau) **2 / sigma**2 )

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
    xhat, yhat = basispursuit(y, F, penalty=0.1)

    plt.figure()
    plt.plot(t,y,'k-', t, yhat, 'r--')
    plt.show()
