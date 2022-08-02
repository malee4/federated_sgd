import numpy as np
from sgd import calculate_local_sgd
import pandas as pd

# sum the local gradients gathered
def aggregateMatrix(local_gradient_set):
    # if nothing is passed in
    if len(local_gradient_set) == 0:
        return None

    aggregated = np.zeros(shape = (len(local_gradient_set[0]), len(local_gradient_set[0][0])))
    
    for matrix in local_gradient_set:
        for j in range(matrix):
            for i in range(matrix[0]):
                aggregated += matrix[j, i]

    return aggregated


# should let this be called if this is the function being called
if __name__ == '__main__':
    # def calculate_local_sgd(M, V, maxiter=100, step=0.1, gamma=0.01, k=10): # add sparse versus ?
    path = ""
    data = pd.read_csv(path)
    
    # clean the data accordingly

    M = np.array([]) # put data into M

    # group into local datasets

    k = 10

    # randomly initialize V
    V = np.random.normal(size = (len(M), k), scale = 1/k)

    local_gradient_set = np.array([])


    for user in M:
        local_gradient = calculate_local_sgd(user, V, k = k)
        curr_num_users = len(local_gradient_set)+1

        local_gradient_set.resize(curr_num_users, len(local_gradient[0]), len(local_gradient[0][0]))
        local_gradient_set[-1:]=local_gradient

