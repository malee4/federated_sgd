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


def update(M, V, maxiter=100, step=0.1, gamma=0.01, k=10):
    local_gradient_set = np.array([])

    # receive all {LG} from all users
    for user in M:
        local_gradient = calculate_local_sgd(user, V, maxiter = maxiter, step = step, gamma = gamma, k = k)
        curr_num_users = len(local_gradient_set)+1

        local_gradient_set.resize(curr_num_users, len(local_gradient[0]), len(local_gradient[0][0]))
        local_gradient_set[-1:]=local_gradient
    
    aggregated_gd = aggregateMatrix(local_gradient_set)

    # update V
    for j in range(len(V)):
        V[j, :] -= step * (aggregated_gd[j, :] + gamma*V[j, :])

    return V


# should let this be called if this is the function being called
if __name__ == '__main__':
    # def calculate_local_sgd(M, V, maxiter=100, step=0.1, gamma=0.01, k=10): # add sparse versus ?
    path = ""
    data = pd.read_csv(path)
    
    # clean the data accordingly
    

    M = np.array([]) # put data into M

    # group into local datasets

    # settings
    k = 10
    maxiter=100 
    step=0.1
    gamma=0.01

    # randomly initialize V
    V = np.random.normal(size = (len(M), k), scale = 1/k)
    
    V = update(M, V)

    


