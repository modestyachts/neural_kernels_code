import numpy as np
import pandas as pd
from enum import Enum

#filenames = ['test_NTK_acc','kernel_acc_new']

#process NTK results, dataset names in names_ntk, acc in acc_ntk
f_ntk = open('test_NTK_acc', "r").readlines()[1:]
acc_ntk = list(map(lambda x: float(x.split()[-1]), f_ntk))
names_ntk = list(map(lambda x: str(x.split()[0]), f_ntk))
bad_idx = []
for i in range(len(acc_ntk)):
    if acc_ntk[i] == 0:
        bad_idx.append(i)
for index in sorted(bad_idx, reverse=True):
    del acc_ntk[index]
    del names_ntk[index]

#process Gaussian kernel results
f_gaussian = open('kernel_acc_new', "r").readlines()[1:]
acc_gaussian = list(map(lambda x: float(x.split()[-1]), f_gaussian))
names_gaussian = list(map(lambda x: str(x.split()[0]), f_gaussian))
for index in sorted(bad_idx, reverse=True):
    del acc_gaussian[index]
    del names_gaussian[index]

#sanity check
if len(acc_ntk) != 90 or len(acc_gaussian) != 90:
    print('num of datasets not equal')

#process ALL accuracy results
print('process old all txt')
f_all = open('acc_all', "r").readlines()
f_all = np.asarray(f_all)
algs_all = f_all[0].split()[1:]
algs_num = len(algs_all)
f_all_new = []
print('total number of classifiers from previous results + min + max + mean + std: ', algs_num)
max_all_acc = []
#top 3: parRF_caret, svm_C, svmPoly_caret
acc_rf = []
acc_svm = []
acc_poly= []
rf_idx = f_all[0].split().index('parRF_caret')
svm_idx = f_all[0].split().index('svm_C')
poly_idx = f_all[0].split().index('svmPoly_caret')
print('idx', rf_idx, svm_idx, poly_idx)

for line in f_all[1:]:
    if line.split()[0] in names_ntk:
        f_all_new.append(line.split())
        max_all_acc.append(float(line.split()[-3]))
        acc_rf.append(float(line.split()[rf_idx]))
        acc_svm.append(float(line.split()[svm_idx]))
        if line.split()[poly_idx] == '--':
            print ('invalid acc for poly at dataset', line.split()[0])
            acc_poly.append(0.0)
        else:
            acc_poly.append(float(line.split()[poly_idx]))

f_all = f_all_new
print('check', len(acc_rf), len(acc_svm), len(acc_poly), len(max_all_acc))

def friedman_ranks(accuracies, our_acc, ntk_acc, include_ours=True):
    acc_df = pd.DataFrame(accuracies[:,1:180], index=accuracies[:,0], columns=algs_all[0:179])
    acc_df["NTK"] = [str(i) for i in ntk_acc]
    if include_ours:
        acc_df["output_kernel"] = [str(i) for i in our_acc]
    acc_df = acc_df.replace(to_replace='--', value=np.nan)
    acc_df = acc_df.astype(float)
    acc_ranks = acc_df.rank(axis=1, ascending=False)
    acc_ranks = acc_ranks.transpose()
    sum_ranks = acc_ranks.sum(axis=1, skipna=True)
    num_ranks = acc_ranks.count(axis=1)
    acc_ranks["avg_ranks"] = sum_ranks / num_ranks
    acc_ranks = acc_ranks["avg_ranks"].to_numpy()
    return acc_ranks

friedman_ranks_with_ours = friedman_ranks(np.asarray(f_all), acc_gaussian, acc_ntk, include_ours=True)
print('Friedman ranks including our output kernel:')
print(f'rf rank: {friedman_ranks_with_ours[rf_idx]}, svm rank: {friedman_ranks_with_ours[svm_idx]}, poly rank: {friedman_ranks_with_ours[poly_idx]}, our rank: {friedman_ranks_with_ours[-1]}, ntk rank: {friedman_ranks_with_ours[-2]}')
friedman_ranks_without_ours = friedman_ranks(np.asarray(f_all), acc_gaussian, acc_ntk, include_ours=False)
print('Friedman ranks excluding our output kernel:')
print(f'rf rank: {friedman_ranks_without_ours[rf_idx]}, svm rank: {friedman_ranks_without_ours[svm_idx]}, poly rank: {friedman_ranks_without_ours[poly_idx]}, ntk rank: {friedman_ranks_without_ours[-1]}')



#compute P90 and P95
ntk90 = 0
ntk95 = 0
ntk_perc = []
g90 = 0
g95 = 0
gaussian_perc = []
rf90 = 0
rf95 = 0
rf_perc = []
svm90 = 0
svm95 = 0
svm_perc = []
poly90 = 0
poly95 = 0
poly_perc = []

for i in range(len(acc_ntk)):
    MAX = max(acc_ntk[i], acc_gaussian[i], max_all_acc[i])
    ntk_perc.append(acc_ntk[i]/MAX)
    gaussian_perc.append(acc_gaussian[i]/MAX)
    rf_perc.append(acc_rf[i]/MAX)
    svm_perc.append(acc_svm[i]/MAX)
    if acc_poly[i] != 0:
        poly_perc.append(acc_poly[i]/MAX)

    if acc_ntk[i] > 0.9 * MAX:
        ntk90 += 1
    if acc_ntk[i] > 0.95 * MAX:
        ntk95 += 1
    if acc_gaussian[i] > 0.9 * MAX:
        g90 += 1
    if acc_gaussian[i] > 0.95 * MAX:
        g95 += 1
    if acc_rf[i] > 0.9 * MAX:
        rf90 += 1
    if acc_rf[i] > 0.95 * MAX:
        rf95 += 1
    if acc_svm[i] > 0.9 * MAX:
        svm90 += 1
    if acc_svm[i] > 0.95 * MAX:
        svm95 += 1
    if acc_poly[i] > 0.9 * MAX:
        poly90 += 1
    if acc_poly[i] > 0.95 * MAX:
        poly95 += 1
P90_NTK = ntk90/90    
P95_NTK = ntk95/90      
P90_GAUSSIAN = g90/90    
P95_GAUSSIAN = g95/90   
P90_RF = rf90/90    
P95_RF= rf95/90   
P90_SVM = svm90/90    
P95_SVM = svm95/90   
P90_POLY = poly90/87    
P95_POLY = poly95/87  
ntk_perc = np.asarray(ntk_perc)    
gaussian_perc = np.asarray(gaussian_perc)  
rf_perc = np.asarray(rf_perc)   
svm_perc = np.asarray(svm_perc)   
poly_perc = np.asarray(poly_perc)     

print('\n')
print('P90 for NTK: ', P90_NTK * 100)
print('P95 for NTK: ', P95_NTK * 100)
print('PMA for NTK: ', np.mean(ntk_perc) * 100)
print('std of PMA for NTK: ', np.std(ntk_perc, ddof=1) * 100)

print('\n')
print('P90 for GAUSSIAN KERNEL: ', P90_GAUSSIAN * 100)
print('P95 for GAUSSIAN KERNEL: ', P95_GAUSSIAN * 100)
print('PMA for GAUSSIAN KERNEL: ', np.mean(gaussian_perc) * 100)
print('std of PMA for GAUSSIAN KERNEL: ', np.std(gaussian_perc, ddof=1) * 100)

print('\n')
print('P90 for RF KERNEL: ', P90_RF * 100)
print('P95 for RF KERNEL: ', P95_RF * 100)
print('PMA for RF KERNEL: ', np.mean(rf_perc) * 100)
print('std of PMA for RF KERNEL: ', np.std(rf_perc, ddof=1) * 100)

print('\n')
print('P90 for SVM GAUSSIAN KERNEL: ', P90_SVM * 100)
print('P95 for SVM GAUSSIAN KERNEL: ', P95_SVM * 100)
print('PMA for SVM GAUSSIAN KERNEL: ', np.mean(svm_perc) * 100)
print('std of PMA for SVM GAUSSIAN KERNEL: ', np.std(svm_perc, ddof=1) * 100)

print('\n')
print('P90 for POLY KERNEL: ', P90_POLY * 100)
print('P95 for POLY KERNEL: ', P95_POLY * 100)
print('PMA for POLY KERNEL: ', np.mean(poly_perc) * 100)
print('std of PMA for POLY KERNEL: ', np.std(poly_perc, ddof=1) * 100)









