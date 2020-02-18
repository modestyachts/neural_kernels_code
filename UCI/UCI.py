import argparse
import os
import math
import numpy as np
import NTK
import kernel
import kernel_relu
import tools
from enum import Enum

Classifier = Enum('Classifier', 'NTK kernel kernel_relu')

parser = argparse.ArgumentParser()
parser.add_argument('-dir', default = "data", type = str, help = "data directory")
parser.add_argument('-file', default = "new_crossvalid_result_10fold.log", type = str, help = "Output File")
parser.add_argument('-max_tot', default = 50, type = int, help = "Maximum num of data points in the dataset") #they used 5000
parser.add_argument('-max_dep', default = 5, type = int, help = "Maximum number of depth")
parser.add_argument('-sample_num', default = 1000, type = int, help = "samples number to estimate squared L2 dist for output kernel")
parser.add_argument('-classifier', default=Classifier.kernel, type = lambda classifier: Classifier[classifier], help='NTK, kernel or kernel_relu')

args = parser.parse_args()

MAX_N_TOT = args.max_tot
MAX_DEP = args.max_dep
sample_num = args.sample_num
DEP_LIST = list(range(MAX_DEP))
datadir = args.dir
classifier = args.classifier

print ('the classifier is', classifier)

alg = tools.svm
avg_acc_list = []
outf = open(args.file, "w")
print ("Dataset\tValidation Acc\tTest Acc", file = outf)
for idx, dataset in enumerate(sorted(os.listdir(datadir))):
    #if dataset != 'ilpd-indian-liver':
        #continue
    #print('dataset found: ', dataset)
    if not os.path.isdir(datadir + "/" + dataset):
        continue
    if not os.path.isfile(datadir + "/" + dataset + "/" + dataset + ".txt"):
        continue
    dic = dict()
    for k, v in map(lambda x : x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v
    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test
    
    #skip invalid dataset
    if n_tot > MAX_N_TOT or n_test > 0:
        print (str(dataset) + '\t0\t0', file = outf)
        continue
    
    #start evaluation for one valid dataset
    print (idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)
    
    # load one valid dataset
    f = open("data/" + dataset + "/" + dic["fich1="], "r").readlines()[1:]
    X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))
    
    # load training and validation set
    fold = list(map(lambda x: list(map(int, x.split())), open(datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
    train_fold, val_fold = fold[0], fold[1]

    if classifier is Classifier.kernel:   
        best_acc = 0.0
        best_value = 0
        best_gamma = 1
        K_best = np.zeros((X.shape[0], X.shape[0]))

        dist_est = kernel.est_dist(X, sample_num)
        print('dist_est for dataset ', dataset, 'is: ', dist_est)

        GAMMA_LIST = [dist_est * (2.0 ** i)for i in range(-19, 20)]
        C_LIST_KERNEL = [2.0 ** i for i in range(-19, 20)]

        for gamma in GAMMA_LIST:
            K = kernel.kernel_value(X, gamma)
            for value in C_LIST_KERNEL:
                print('gamma ', gamma, 'c', value, 'dataset ', idx, dataset)
                acc = alg(K[train_fold][:, train_fold], K[val_fold][:, train_fold], y[train_fold], y[val_fold], value, c)
                all_accs = [acc]
                for i in range(10):
                    idxs = np.hstack([train_fold,val_fold])
                    #class_balanced
                    new_train_fold = []
                    new_val_fold = []
                    for c in range(np.max(y)+1):
                        all_idxs = np.where(y==c)[0]
                        np.random.shuffle(all_idxs)
                        new_train_fold = np.append(new_train_fold, all_idxs[0:int(len(all_idxs)/2)])
                        new_val_fold = np.append(new_val_fold, all_idxs[int(len(all_idxs)/2):])
                    new_train_fold = new_train_fold.astype(int)
                    new_val_fold = new_val_fold.astype(int)
                    new_acc = alg(K[new_train_fold][:, new_train_fold], K[new_val_fold][:, new_train_fold], y[new_train_fold], y[new_val_fold], value, c)
                    all_accs.append(new_acc)
                acc = np.mean(all_accs)
                if acc > best_acc:
                    best_acc = acc
                    best_value = value
                    best_gamma = gamma
                    K_best = K
        print('get best K')
        K = K_best
        print ("best acc:", best_acc, "\tbest C:", best_value, "\tbest gamma:", best_gamma)
        
    if classifier is Classifier.kernel_relu:   
        C_LIST_KERNEL_RELU = [2.0 ** i for i in range(-8, 15)]
        best_acc = 0.0
        best_value = 0
        K_best = np.zeros((X.shape[0], X.shape[0])) 
        K = kernel_relu.kernel_value(X)
        for value in C_LIST_KERNEL_RELU:
            acc = alg(K[train_fold][:, train_fold], K[val_fold][:, train_fold], y[train_fold], y[val_fold], value, c)
            if acc > best_acc:
                best_acc = acc
                best_value = value
                K_best = K
        K = K_best
        print ("best acc:", best_acc, "\tbest C:", best_value)
    

    if classifier is Classifier.NTK:
        C_LIST = [2.0 ** i for i in range(-19, 20)]
        Ks = NTK.kernel_value_batch(X, MAX_DEP)
        best_acc = 0.0
        best_value = 0
        best_dep = 0
        best_ker = 0
        
        # enumerate kernels and cost values to find the best hyperparameters
        for dep in DEP_LIST:
            for fix_dep in range(dep + 1):
                print('start tune dep/fix dep ', dep, fix_dep, 'for dataset ', idx, dataset)
                K = Ks[dep][fix_dep]
                for value in C_LIST:
                    acc = alg(K[train_fold][:, train_fold], K[val_fold][:, train_fold], y[train_fold], y[val_fold], value, c)
                    all_accs = [acc]
                    for i in range(4):
                        idxs = np.hstack([train_fold,val_fold])
                        #class_balanced
                        new_train_fold = []
                        new_val_fold = []
                        for c in range(np.max(y)+1):
                            all_idxs = np.where(y==c)[0]
                            np.random.shuffle(all_idxs)
                            new_train_fold = np.append(new_train_fold, all_idxs[0:int(len(all_idxs)/2)])
                            new_val_fold = np.append(new_val_fold, all_idxs[int(len(all_idxs)/2):])
                        new_train_fold = new_train_fold.astype(int)
                        new_val_fold = new_val_fold.astype(int)
                        new_acc = alg(K[new_train_fold][:, new_train_fold], K[new_val_fold][:, new_train_fold], y[new_train_fold], y[new_val_fold], value, c)
                        all_accs.append(new_acc)
                    acc = np.mean(all_accs)
                    if acc > best_acc:
                        best_acc = acc
                        best_value = value
                        best_dep = dep
                        best_fix = fix_dep
        
        K = Ks[best_dep][best_fix]
        print ("best acc:", best_acc, "\tbest C:", best_value, "\t best dep:", best_dep, "\tbest fix dep:", best_fix)
    
    # 4-fold cross-validating
    avg_acc = 0.0
    fold = list(map(lambda x: list(map(int, x.split())), open("data/" + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))
    for repeat in range(4):
        train_fold, test_fold = fold[repeat * 2], fold[repeat * 2 + 1]
        acc = alg(K[train_fold][:, train_fold], K[test_fold][:, train_fold], y[train_fold], y[test_fold], best_value, c)
        avg_acc += 0.25 * acc

        
    print ("acc:", avg_acc, "\n")
    print (str(dataset) + '\t' + str(best_acc * 100) + '\t' + str(avg_acc * 100), file = outf)
    avg_acc_list.append(avg_acc)

print ("avg_acc:", np.mean(avg_acc_list) * 100)
print ("std", np.std(avg_acc_list, ddof=1) * 100) #sample std
outf.close()

    
    
