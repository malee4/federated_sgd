################################################################################
# This program has been adapted.
# Authors: Hai Lan, Oden Institute for Computational Engineering and Science
# Title: "Matrix Completion Using Stochastic Gradient Descent"
# GitHub Link: https://github.com/lawrencelan97/sgd
# Paper Published: Spring 2019
################################################################################
import numpy as np


# create U0, V0, where U = users, V = ratings
# note V is downloaded at the start
def calculate_local_sgd(M, V, maxiter=100, step=0.1, gamma=0.01, k=10, created_sparse_data = False, actual_data = []): # add sparse versus 
    # randomly initialize U locally
    U = np.random.normal(size = (len(M), k), scale = 1/k)

    # to be returned
    local_gradient = np.zeros(shape = (len(M), len(M[0]))) # what should the dimensions be?

    for itr in range(maxiter):
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


    return local_gradient
