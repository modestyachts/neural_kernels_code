import numpy as np
import scipy.linalg
import sklearn.metrics as metrics


def train_ls_dual_model(K_train, y_train, reg):
    K_train = K_train.astype('float64')
    y = np.eye(np.max(y_train) + 1)[y_train]
    idxs = np.diag_indices(K_train.shape[0])
    K_train[idxs] += reg
    model = scipy.linalg.solve(K_train, y, sym_pos=True)
    K_train[idxs] -= reg
    return model, 0.0

def train_ls_dual_model_loo_tilt(K_train, y_train, K_test, y_test, reg):
    K_train = K_train.astype('float64')
    y = np.eye(np.max(y_train) + 1)[y_train]
    idxs = np.diag_indices(K_train.shape[0])
    K_train[idxs] += reg
    K_train_inv = np.linalg.inv(K_train)
    K_train[idxs] -= reg
    alpha_regular = K_train_inv.dot(y)
    Q = np.diag(K_train_inv)[:, np.newaxis]
    y_loo = y - Q/alpha
    results = []
    for tilt_fac in range(100):
        tilt_fac /= 100
        alpha_loo_tilt = K_train_inv.dot(y - tilt_fac*y_loo)
        y_test_pred = np.argmax(K_test.dot(alpha_loo_tilt), axis=1)
        acc_test = metrics.accuracy_score(y_test_pred, y_test)
        results.append((acc_test, alpha_loo_tilt))
    best_acc, model = max(results, key_func=lambda x: x[0])
    return model, 0.0

def train_ls_dual_model_center(K_train, y_train, reg):
    K_train = K_train.astype('float64')
    K_row_sums = np.sum(K_train, axis=1)[:, np.newaxis]

    K_train_c = K_train.copy()
    K_train_c -= K_row_sums/K_train.shape[0]
    K_train_c -= K_row_sums.T/K_train.shape[0]
    K_train_c += np.sum(K_train)/(K_train.shape[0]*K_train.shape[0])

    y = np.eye(np.max(y_train) + 1)[y_train]
    y_c = y - np.mean(y, axis=0)

    idxs = np.diag_indices(K_train.shape[0])
    K_train_c[idxs] += reg
    model = scipy.linalg.solve(K_train_c, y_c, sym_pos=True)
    K_train_c[idxs] -= reg
    bias = np.sum((y - K_train.dot(model)), axis=0)/K_train.shape[0]
    return model, bias


def eval_ls_model(model, bias, K, y):
    K = K.astype('float64')
    logits = K.dot(model) + bias
    preds = np.argmax(logits, axis=1)
    return logits, preds, np.sum(preds == y)/y.shape[0]
