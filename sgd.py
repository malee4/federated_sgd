################################################################################
# Authors: Hai Lan, Oden Institute for Computational Engineering and Science
# Title: "Matrix Completion Using Stochastic Gradient Descent"
# GitHub Link: https://github.com/lawrencelan97/sgd
# Paper Published: Spring 2019
################################################################################

# import necessary libraries
import pandas as pd
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import norm
import random
import matplotlib.pyplot as plt
import time

# compute mean squared error: NORM(UV-M, 'fro')
def MSE(U, V, M, obs):
    s = 0
    st = 0
    nSamp = len(obs)
    for l in range(nSamp):
        i, j = obs[l]
        s += (np.dot(U[i, :], V[j, :].T) - M[i, j])**2
    return np.sqrt(s)

# compute relative Frobenius error: NORM(UV-M, 'fro')/NORM(M, 'fro')
def RelError(U, V, M, obs):
    s = 0
    st = 0
    nSamp = len(obs)
    for l in range(nSamp):
        i, j = obs[l]
        s += (np.dot(U[i, :], V[j, :].T) - M[i, j])**2
        st += (M[i, j])**2
    return np.sqrt(s)/np.sqrt(st)