# import necessary libraries import pandas as pd
from unicodedata import ucd_3_2_0
import numpy as np
from pandas import NA
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import norm
import random
import matplotlib.pyplot as plt 
import time


M = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]) # sample dataset
# print(M)

max_iterations = 50
step = 0.1
gamma = 0.01
k=1
# randomly initialize U and V
U = np.random.normal(size = (len(M), k), scale = 1/k)
V = np.random.normal(size = (len(M[0]), k), scale = 1/k)

print(U)
print(V)


local_gradient = np.zeros(shape = (len(M), len(M[0]))) # what should the dimensions be?

for itr in range(max_iterations):
    for i in range(len(M)):
        for j in range(len(M[0])):
             # compute difference
            difference = np.dot(U[i, :], V[j, :].T)

            # if data exists at i,j
            if not M[i, j]:
                continue
            # complete partial calculation
            error = difference - M[i, j]

            # update U
            U[i, :] -= step * (error*V[j, :] + gamma*U[i, :])

            # calculate some component of the gradient for V
            local_gradient[i, j] = error * U[i, :] + gamma * V[i, :]
print(local_gradient)






