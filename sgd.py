################################################################################
# This program has been adapted.
# Authors: Hai Lan, Oden Institute for Computational Engineering and Science
# Title: "Matrix Completion Using Stochastic Gradient Descent"
# GitHub Link: https://github.com/lawrencelan97/sgd
# Paper Published: Spring 2019
################################################################################


# import packages
import numpy as np
from linear_tools import MSE, RelError

# create U0, V0, where U = users, V = ratings
# note V is downloaded at the start
def calculate_local_sgd(data, V, maxiter=100, alpha=0.1, beta=0.01, k=10, created_sparse_data = False, actual_data = []): # add sparse versus 
    # if created_sparse_data and not data:
    #     raise Exception("Please include the original dataset")
    # elif created_sparse_data:
    #     # create error vector
    #     errorVec = np.zeros((2, maxiter), dtype='float')
    #     RelErr_prev = 1
    
    # conduct preliminary setup
    N = data.length # matrix size
    U = np.random.normal(size=(N, k), scale=1/k)

    # remove this and replace with download V
    # V = np.random.normal(size=(N, k), scale=1/k) 
    
    # for every iteration in amount specified
    for iteration in maxiter:
        # randomize data, select 10% of items
        np.random.shuffle(data)
        selected_updates = data[1:(int)(np.floor(0.1*len(data)))]

        for l in selected_updates:
            i, j = l
            error = ( np.dot(U[i, :], V[j, :].T) - data[i, j] ) 
            U_temp = np.copy(U[i, :])
            # update U, equation <1> 
            U[i, :] -= alpha * (error*V[j, :] + beta*U[i, :])
            
            # update V
            V[j, :] -= alpha * (error*U_temp + beta*V[j, :]) # UNCERTAIN: this update function doesn't reflect the mathematics
    return V



    
    
    



    # for each user's movie preferences (V, userData):

        # calculate partial of users 

        #
