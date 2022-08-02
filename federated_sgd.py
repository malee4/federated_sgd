import numpy as np

# clear up data

def aggregateMatrix(LG_set = np.array([])):
    # if nothing is passed in
    if LG_set.size == 0:
        return None
    V = np.zeros(shape = (LG[0].size, LG[0][0].size))

    for LG in LG_set:
        for i in LG.size:
            for j in LG[0].size:
                V[i, j] += LG[i, j]

    # UNCERTAIN: get the average? (aggregate)
    for i in V.size:
        for j in V[0].size:
            V[i, j] = V[i, j] / LG_set.size

    return V

def updateV(LG, V, alpha=0.1, beta=0.01):
    if V.size == 0:
        return None
    
    for i in V.size:
        for j in V[0].size:
