import math
import numpy as np
import random
import time

# return an array K of size (N, N), K[i][j] is kernel value 
def kernel_value(X):
    #print('start compute output kernel for ntk')
    #print('gamma value is ', gamma)
    #starttime = time.time()
    K = X.dot(X.T)
    x_norms= np.linalg.norm(X, axis=1)[:, np.newaxis]
    K = K / x_norms / x_norms.T
    K = np.clip(K, -1.0, 1.0)
    theta = np.arccos(K)
    K_out = (np.sin(theta) + (np.pi - theta) * K)
    K_out *= x_norms
    K_out *= x_norms.T
    K_out /= np.pi
    #print('finish computing output kernel for ntk')
    #runtime = time.time()-starttime
    #print('took ',runtime, 'seconds')
    return K
