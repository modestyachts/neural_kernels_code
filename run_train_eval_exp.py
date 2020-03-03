
import multiprocessing
try:
    multiprocessing.set_start_method("spawn")
except:
    print("Context already set..")

import argparse
import io
import json
import pickle
import sys
import random
from timeit import default_timer as timer

import dill
import numpy as np
import scipy.linalg
from functools import partial
import torch
import PIL
from PIL import Image

import kernel_gen
import utils
from config import arch_short_name, get_cfg_defaults, hash_arch
from ls import eval_ls_model, train_ls_dual_model, train_ls_dual_model_center
from sklearn import metrics

def main():
    parser = argparse.ArgumentParser("Train + evaluate kernel model")
    parser.add_argument("--config", default=None)
    parser.add_argument('--config-updates', default=[], nargs='*')

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    if args.config is not None:
        cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.config_updates)
    if cfg.SYSTEM.NUM_GPUS == 0:
        cfg.SYSTEM.NUM_GPUS = torch.cuda.device_count()
    cfg.freeze()
    print(cfg)

    # load dataset
    random.seed(cfg.DATASET.RAND_SEED)

    X_train, y_train, X_test, y_test = load_dataset(cfg)
    print(X_train.shape)
    print(X_test.shape)
    K_train, K_test = generate_kernels(cfg, X_train, X_test)
    best_model_result, best_train_result, best_test_result = solve_kernel(cfg, K_train, y_train, K_test, y_test)

def load_dataset(cfg):
    print("Loading dataset...")
    dataset = utils.load_dataset(cfg.DATASET.NAME)

    X_train = dataset['X_train'].astype('float64')
    y_train = dataset['y_train']

    X_test = dataset['X_test'].astype('float64')
    y_test = dataset['y_test']

    if cfg.SYSTEM.FLOAT_32:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

    if cfg.DATASET.TRAIN_SUBSET is not None:
        if cfg.DATASET.RANDOM_TRAIN_SUBSET:
            np.random.seed(cfg.DATASET.RAND_SEED)
            train_idxs = np.random.choice(
                cfg.DATASET.TRAIN_SUBSET, X_train.shape[0], replace=False)
        else:
            train_idxs = list(np.arange(cfg.DATASET.TRAIN_SUBSET))
        X_train = X_train[train_idxs]
        y_train = y_train[train_idxs]
    else:
        train_idxs = list(np.arange(X_train.shape[0]))

    if cfg.DATASET.TEST_SUBSET is not None:
        if cfg.DATASET.RANDOM_TEST_SUBSET:
            np.random.seed(cfg.DATASET.RAND_SEED)
            test_idxs = np.random.choice(
                cfg.DATASET.TEST_SUBSET, X_test.shape[0], replace=False)
        else:
            test_idxs = list(np.arange(0, cfg.DATASET.TEST_SUBSET))
        X_test = X_test[test_idxs]
        y_test = y_test[test_idxs]
    else:
        test_idxs = list(np.arange(X_test.shape[0]))
    return X_train, y_train, X_test, y_test

def generate_kernels(cfg, X_train, X_test, extra_info=None, train_eval_id=None, test_eval_id=None):
    if extra_info is None:
        extra_info = {}

    d_net = kernel_gen.GenericKernel(kernel_cfg=cfg.KERNEL, cache_path=cfg.SYSTEM.CACHE_PATH, float32=cfg.SYSTEM.FLOAT_32)
    print("Generating train kernel...")
    print(f"using {cfg.SYSTEM.NUM_GPUS} gpus")
    train_start = timer()
    K_train = kernel_gen.generate_kernel_parallel(cfg.KERNEL, X_train, X_train, num_gpus=cfg.SYSTEM.NUM_GPUS, symmetric=True, batch_size=cfg.SYSTEM.BATCH_SIZE, cache_path=cfg.SYSTEM.CACHE_PATH, float32=cfg.SYSTEM.FLOAT_32, extra_info={"kernel_type": "Train"})
    train_end = timer()
    print(f"{X_train.shape[0]} x {X_train.shape[0]} Train kernel took {train_end - train_start}")
    print("Generating test kernel...")
    test_start = timer()
    K_test = kernel_gen.generate_kernel_parallel(cfg.KERNEL, X_test, X_train, num_gpus=cfg.SYSTEM.NUM_GPUS, batch_size=cfg.SYSTEM.BATCH_SIZE, cache_path=cfg.SYSTEM.CACHE_PATH, float32=cfg.SYSTEM.FLOAT_32, verbose=cfg.SYSTEM.VERBOSE)
    if cfg.SYSTEM.FLOAT_32:
        K_train = K_train.astype('float64')
        K_test  = K_test.astype('float64')
    return K_train, K_test

def solve_kernel(cfg, K_train, y_train, K_test, y_test):
    print("Solving Kernel....")
    all_results = []
    # do a small sweep by default
    for reg in cfg.SOLVE.REGS:
        try:
            model_result = {}
            test_result = {}
            train_result = {}
            if cfg.SOLVE.LOO_TILT:
                model, bias = train_ls_dual_model_loo_tilt(K_train, y_train, K_test, y_test, reg)
            else:
                model, bias = train_ls_dual_model(K_train, y_train, reg)
            model_result["model"] = model
            model_result["bias"] = bias
            model_result["reg"] = reg
            model_result["kernel_cfg"] = cfg.KERNEL

            train_logits, train_preds, train_acc = eval_ls_model(
                model, bias, K_train, y_train)
            train_result["logits"] = train_logits
            train_result["preds"] = train_preds
            train_result["acc"] = train_acc
            train_result["kernel"] = K_train

            test_logits, test_preds, test_acc = eval_ls_model(
                model, bias, K_test, y_test)
            test_result["kernel"] = K_test
            test_result["logits"] = test_logits
            test_result["preds"] = test_preds
            test_result["acc"] = test_acc

            print(f"\tReg: {reg}")
            print(f"\tTrain Accuracy: {train_acc}")
            print(f"\tTest Accuracy: {test_acc}")
            all_results.append((model_result, train_result, test_result))
        except scipy.linalg.LinAlgError:
            print("\tregularizer error: ", reg)

    best_model_result, best_train_result, best_test_result = max(all_results, key=lambda x: x[2]['acc'])

    return best_model_result, best_train_result, best_test_result




if __name__ == "__main__":
    main()
