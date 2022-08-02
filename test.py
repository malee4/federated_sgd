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
V = np.random.normal(size = (k, len(M[0])), scale = 1/k)

print(U)
print(V)

for i in M: 
    print(i)


for itr in range(max_iterations):
    for i in range(len(M)):
        for j in range(len(M[0]))




