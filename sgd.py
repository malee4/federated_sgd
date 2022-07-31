# import packages
import numpy as np
from tools import MSE, RelError

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


    
    
    



    # for each user's movie preferences (V, userData):

        # calculate partial of users 

        #
