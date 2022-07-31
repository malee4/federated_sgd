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

### TEST 1: MOVIELENS ###

# read in the movielens dataset https://www.kaggle.com/grouplens/movielens-latest-small
dataset = pd.read_csv('ratings.csv').drop('timestamp', axis=1)
print(dataset.shape)

# convert to numpy matrix
M = dataset.pivot(index='userId', columns='movieId', values='rating').values
print(M.shape)

# produce vectors of users, movies, ratings (given values)
users = dataset['userId'].values
movies = dataset['movieId'].values
ratings = dataset['rating'].values
obs = [(users[i]-1, movies[i]-1) for i in range(len(users))] # form tuples

# form sparse matrix
M = sparse.coo_matrix((ratings, (users-1,movies-1))).tocsr()
print(M)

# set up parameters for SGD
maxiter = 50

alpha = 0.1 # step size (learning rate)
beta = 0.01 # regularization

nSamp = len(obs) # size of the observed set

np.random.shuffle(obs)

# form training and test sets (hyperparameter)
train = obs[1:(int)(np.floor(0.7*nSamp))] # training set [70%]
test = obs[(int)(np.floor(0.7*nSamp)):len(obs)+1] # test set [30%]

start = time.time()

# start SGD

k = 10 # guess of the rank of matrix

# initialize matrices
U = np.random.normal(size=(np.max(users), k), scale=1/k)
V = np.random.normal(size=(np.max(movies), k), scale=1/k)

errorVec = np.zeros((2, maxiter), dtype='float')

RelErr_prev = 1

# for every iteration 
for itr in range(maxiter):
    np.random.shuffle(train)
    selected_updates = train[1:(int)(np.floor(0.1*len(train)))] # batch size (hyperparameter) [10%]
    
    for l in selected_updates:
        i, j = l
        error = ( np.dot(U[i, :], V[j, :].T) - M[i, j] )
        U_temp = np.copy(U[i, :])
        # update U
        U[i, :] -= alpha * (error*V[j, :] + beta*U[i, :])
        # update V
        V[j, :] -= alpha * (error*U_temp + beta*V[j, :])

    errorVec[0, itr] = MSE(U, V, M, test)
    errorVec[1, itr] = RelError(U, V, M, test)

    # print errors during iteration
    if RelErr_prev - errorVec[1, itr] < 0.001:
        break
    else:
        print("iter", itr+1,
              ": MSE =", errorVec[0, itr],
              ", RelErr =", errorVec[1, itr])
        RelErr_prev = errorVec[1, itr]

end = time.time()
print("\nRunning time:", end-start)

# print matrices
print("True Answer")
print(M)
print("\nEstimated Answer")
print(np.dot(U, V.T))
print("\nAccuracy (estimated - true)")
print(np.dot(U, V.T)-M)

# plot error curves for maxiter
plt.plot(errorVec[0, :])
plt.title('Absolute Error vs Iteration')
plt.xlabel('Iteration #')
plt.ylabel('Absolute Error')
plt.show()

plt.plot(errorVec[1, :])
plt.title('Relative Error vs Iteration')
plt.xlabel('Iteration #')
plt.ylabel('Relative Error')
plt.show()

### TEST 2: MATRIX FROM WELL-SEPARATED KERNEL ###

# construct a low-rank matrix
N = 1000  # matrix size

d = 1 # dimension of point set
# generate well-separated point set
X = np.random.randn(d, N) + 5 # [the larger the distance, the smaller the rank]
Y = np.random.randn(d, N)  

bandwidth = 10 # bandwidth of Gaussian kernel [the larger the bandwidth, the smaller the rank]
M = np.zeros([N, N]) # fill in full true matrix M
for i in range(N):
    for j in range(N):
        M[i, j] = np.exp(-np.linalg.norm(X[:, i]-Y[:, j])**2/bandwidth)

print(M)
print("\nRank of M:", np.linalg.matrix_rank(M))

col_idx = [np.random.randint(1, N) for i in range(100000)]
row_idx = [np.random.randint(1, N) for i in range(100000)]
obs = [(i, j) for i, j in zip(row_idx, col_idx)]

# set up parameters for SGD
maxiter = 100

alpha = 0.1 # step size (learning rate)
beta = 0.01 # regularization

nSamp = len(obs) # size of the observed set

np.random.shuffle(obs)

# form training and test sets (hyperparameter)
train = obs[1:(int)(np.floor(0.7*nSamp))] # training set [70%]
test = obs[(int)(np.floor(0.7*nSamp)):len(obs)+1] # test set [30%]

# start SGD

k = 10 # guess of the rank of matrix

# initialize matrices
U = np.random.normal(size=(N, k), scale=1/k)
V = np.random.normal(size=(N, k), scale=1/k)

errorVec = np.zeros((2, maxiter), dtype='float')

for itr in range(maxiter):
    np.random.shuffle(train)
    selected_updates = train[1:(int)(np.floor(0.1*len(train)))] # batch size (hyperparameter) [10%]
    
    for l in selected_updates:
        i, j = l
        error = ( np.dot(U[i, :], V[j, :].T) - M[i, j] )
        U_temp = np.copy(U[i, :])
        # update U
        U[i, :] -= alpha * (error*V[j, :] + beta*U[i, :])
        # update V
        V[j, :] -= alpha * (error*U_temp + beta*V[j, :])

    errorVec[0, itr] = MSE(U, V, M, test)
    errorVec[1, itr] = RelError(U, V, M, test)

    # print errors during iteration
    print("iter", itr+1,
          ": MSE =", errorVec[0, itr],
          ", RelErr =", errorVec[1, itr])

# print matrices
print("True Answer")
print(M)
print("\nEstimated Answer")
print(np.dot(U, V.T))
print("\nAccuracy (estimated - true)")
print(np.dot(U, V.T)-M)

### TEST 3: RANK-1 MATRIX ###

# construct a low-rank matrix
N = 1000  # matrix size

# random normalized vectors
vec1 = np.random.randn(1, N)
vec2 = np.random.rand(N)
vec1 = vec1/np.linalg.norm(vec1)
vec2 = vec2/np.linalg.norm(vec2)

M = np.outer(vec1, vec2)

print(M)
print("\nRank of M:", np.linalg.matrix_rank(M))

col_idx = [np.random.randint(1, N) for i in range(100000)]
row_idx = [np.random.randint(1, N) for i in range(100000)]
obs = [(i, j) for i, j in zip(row_idx, col_idx)]

# set up parameters for SGD
maxiter = 100

alpha = 0.1 # step size (learning rate)
beta = 0.01 # regularization

nSamp = len(obs) # size of the observed set

np.random.shuffle(obs)

# form training and test sets (hyperparameter)
train = obs[1:(int)(np.floor(0.7*nSamp))] # training set [70%]
test = obs[(int)(np.floor(0.7*nSamp)):len(obs)+1] # test set [30%]

# start SGD

k = 10 # guess of the rank of matrix

# initialize matrices
U = np.random.normal(size=(N, k), scale=1/k)
V = np.random.normal(size=(N, k), scale=1/k)

errorVec = np.zeros((2, maxiter), dtype='float')

for itr in range(maxiter):
    np.random.shuffle(train)
    selected_updates = train[1:(int)(np.floor(0.1*len(train)))] # batch size (hyperparameter) [10%]
    
    for l in selected_updates:
        i, j = l
        error = ( np.dot(U[i, :], V[j, :].T) - M[i, j] )
        U_temp = np.copy(U[i, :])
        # update U
        U[i, :] -= alpha * (error*V[j, :] + beta*U[i, :])
        # update V
        V[j, :] -= alpha * (error*U_temp + beta*V[j, :])

    errorVec[0, itr] = MSE(U, V, M, test)
    errorVec[1, itr] = RelError(U, V, M, test)

    # print errors during iteration
    print("iter", itr+1,
          ": MSE =", errorVec[0, itr],
          ", RelErr =", errorVec[1, itr])

# print matrices
print("True Answer")
print(M)
print("\nEstimated Answer")
print(np.dot(U, V.T))
print("\nAccuracy (estimated - true)")
print(np.dot(U, V.T)-M)

### TEST 4: LOW-RANK MATRIX WITH HIGH CONDITION NUMBER ###

# construct a low-rank matrix
N = 1000  # matrix size

tempM = np.random.randn(N,N)

U,S,V = np.linalg.svd(tempM)

# manipulate S to ensure low-rank
S[0] = 100
S[19] = 0.01
for i in range(20,N):
    S[i] = 0

M = U*S*V.T

print(M)
print("\nRank of M:", np.linalg.matrix_rank(M))
print("Condition number of M:", np.linalg.cond(M))

col_idx = [np.random.randint(1, N) for i in range(100000)]
row_idx = [np.random.randint(1, N) for i in range(100000)]
obs = [(i, j) for i, j in zip(row_idx, col_idx)]

# set up parameters for SGD
maxiter = 100

alpha = 0.1 # step size (learning rate)
beta = 0.01 # regularization

nSamp = len(obs) # size of the observed set

np.random.shuffle(obs)

# form training and test sets (hyperparameter)
train = obs[1:(int)(np.floor(0.7*nSamp))] # training set 70%
test = obs[(int)(np.floor(0.7*nSamp)):len(obs)+1] # test set 30%

# start SGD

k = 20 # guess of the rank of matrix

# initialize matrices
U = np.random.normal(size=(N, k), scale=1/k)
V = np.random.normal(size=(N, k), scale=1/k)

errorVec = np.zeros((2, maxiter), dtype='float')

for itr in range(maxiter):
    np.random.shuffle(train)
    selected_updates = train[1:(int)(np.floor(0.1*len(train)))] # batch size (hyperparameter)
    
    for l in selected_updates:
        i, j = l
        error = ( np.dot(U[i, :], V[j, :].T) - M[i, j] )
        U_temp = np.copy(U[i, :])
        # update U
        U[i, :] -= alpha * (error*V[j, :] + beta*U[i, :])
        # update V
        V[j, :] -= alpha * (error*U_temp + beta*V[j, :])

    errorVec[0, itr] = MSE(U, V, M, test)
    errorVec[1, itr] = RelError(U, V, M, test)

    # print errors during iteration
    print("iter", itr+1,
          ": MSE =", errorVec[0, itr],
          ", RelErr =", errorVec[1, itr])

# print matrices
print("True Answer")
print(M)
print("\nEstimated Answer")
print(np.dot(U, V.T))
print("\nAccuracy (estimated - true)")
print(np.dot(U, V.T)-M)