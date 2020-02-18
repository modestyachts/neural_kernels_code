import math
import numpy as np
import random
import time

# return an array K of size (N, N), K[i][j] is kernel value 
def kernel_value(X, gamma):
    #print('start compute output kernel')
    #print('gamma value is ', gamma)
    #starttime = time.time()
    K = X.dot(X.T)
    norms_sq = (np.linalg.norm(X,axis=1)[:,np.newaxis])**2
    K = np.exp(-gamma*(norms_sq - 2*K + norms_sq.T))
    #print('finish computing output kernel')
    #runtime = time.time()-starttime
    #print('took ',runtime, 'seconds')
    return K

#estimate the sq L2 distance between two data points for a given X in N by d
def est_dist(X, sample_num):
    dist_est = 1
    dist_list = []
    
    if X.shape[0] <= sample_num:
        for i in range(X.shape[0]):
            for j in range(i+1, X.shape[0]):
                dist_sq = (np.linalg.norm(X[i]-X[j]))**2
                dist_list.append(dist_sq)
    else:
        idx = random.sample(range(X.shape[0]), sample_num)
        for i in range(sample_num):
            for j in range(i+1, sample_num):
                dist_sq = (np.linalg.norm(X[idx[i]]-X[idx[j]]))**2
                dist_list.append(dist_sq)

    dist_list = np.asarray(dist_list) 
    dist_est = np.median(dist_list)
    return dist_est

#insert a midpoint between each pair of entries of an np array
def insert_midpoint(X):
    X = np.asarray(X)
    mid_array = (X[1:] + X[:-1]) / 2.0
    c = np.empty((X.size + X.size-1,), dtype=X.dtype)
    c[0::2] = X
    c[1::2] = mid_array
    return c


