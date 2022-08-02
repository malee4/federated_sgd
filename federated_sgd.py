import numpy as np
from sgd import calculate_local_sgd

# clear up data

def aggregateMatrix(local_gradient_set):
    # if nothing is passed in
    if len(local_gradient_set) == 0:
        return None

    aggregated = np.zeros(shape = (len(local_gradient_set[0]), len(local_gradient_set[0][0])))
    
    for matrix in local_gradient_set:
        for i in range(matrix):
            for j in range(matrix[0]):
                aggregated += matrix[i, j]


    return

def updateV(LG, V, alpha=0.1, beta=0.01):
    if LG.size == 0:
        return 
    elif LG[0].size == 0:
        return

    V[j, :] -= alpha * (error*U_temp + beta*V[j, :])

