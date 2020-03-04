import run_train_eval_exp

import argparse
import boto3
import contextlib
import io
import itertools
import json
import pickle
import urllib
from collections import defaultdict

import click
import dateparser
import dill
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
import seaborn as sns
import yacs
from PIL import Image
from sklearn import metrics
import torch
import torchvision

import augmentation_transforms
import random_aug
import utils
import yaml
from config import arch_short_name, get_cfg_defaults, hash_arch, CN

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010])

@click.group("preprocess")
def preprocess():
    pass


@preprocess.command("cifar100")
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/myrtle11_exp.yaml")
def preprocess_cifar100(config_updates, config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)
    cfg_dict = utils.cfg_to_dict(cfg)
    trainset = torchvision.datasets.CIFAR100('./data', train=True, transform=None, target_transform=None, download=True)
    testset = torchvision.datasets.CIFAR100('./data', train=False, transform=None, target_transform=None, download=True)
    X_train = trainset.data
    X_test = testset.data
    N_train = len(trainset.data)
    N_test = len(testset.data)
    train_idxs = np.arange(N_train)
    test_idxs = np.arange(N_test)
    X_train = X_train[train_idxs]
    X_test = X_test[test_idxs]
    y_train = np.array(trainset.targets)[train_idxs]
    y_test = np.array(testset.targets)[test_idxs]
    X_train_full = [X_train]
    y_train_full = [y_train]
    for i,(aug_name, mag) in enumerate(cfg.DATASET.AUGMENTATIONS):
        print(f"Applying augmentation: {aug_name}, {mag}")
        if aug_name  == "Random":
            aug_transformer = random_aug.randaugment(cfg.DATASET.RAND_AUGMENT_N_MAX, mag, seed=cfg.DATASET.RAND_SEED+i, transforms=cfg.DATASET.RAND_AUGMENT_AUGS, replace=cfg.DATASET.RAND_AUGMENT_REPLACE, random_n=cfg.DATASET.RAND_AUGMENT_RAND_N)
        else:
            aug_transformer = augmentation_transforms.NAME_TO_TRANSFORM[aug_name].pil_transformer(1.0, mag, X_train[0].shape)
        X_train_aug  = utils.apply_transform(aug_transformer, X_train)
        X_train_full.append(X_train_aug)
        y_train_full.append(y_train)

    X_train_full = np.vstack(X_train_full)
    y_train_full = np.hstack(y_train_full)
    (X_train, X_test), global_ZCA = utils.preprocess((X_train_full/255).astype('float32'), (X_test/255).astype('float32'), min_divisor=1e-8, zca_bias=1e-4, return_weights=True)


    print('Got Data')


    train_bio = io.BytesIO()
    test_bio = io.BytesIO()

    np.savez("cifar100_train.npz", X_train=X_train, y_train=y_train, global_ZCA=global_ZCA)
    np.savez("cifar100_test.npz", X_test=X_test, y_test=y_test, global_ZCA=global_ZCA)



@preprocess.command("cifar10")
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/myrtle11_exp.yaml")
def preprocess_cifar(config_updates, config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)
    cfg_dict = utils.cfg_to_dict(cfg)


    # load cifar-10 from db
    data = utils.load_dataset("cifar-10-raw")
    y_test_full = data["y_test"]
    X_test_raw = data["X_test"]
    y_train_full = data["y_train"]
    X_train_raw = data["X_train"]

    # load cifar-10.1
    X_test_2 = np.load(io.BytesIO(urllib.request.urlopen("https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy").read()))
    y_test_2_full = np.load(io.BytesIO(urllib.request.urlopen("https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy").read()))


    cfg.SYSTEM.NUM_GPUS = torch.cuda.device_count()

    N_train = cfg.DATASET.TRAIN_SUBSET
    N_test = cfg.DATASET.TEST_SUBSET

    np.random.seed(cfg.DATASET.RAND_SEED)
    if cfg.DATASET.RANDOM_TRAIN_SUBSET:
        train_idxs = np.random.choice(X_train_raw.shape[0], N_train, replace=False)
    else:
        train_idxs = np.arange(N_train)

    if cfg.DATASET.RANDOM_TEST_SUBSET:
        test_idxs = np.random.choice(X_test_raw.shape[0], N_test, replace=False)
    else:
        test_idxs = np.arange(N_test)

    print(f"train_idxs: {train_idxs}")
    print(f"test_idxs: {test_idxs}")
    X_train = X_train_raw[train_idxs]
    X_test = X_test_raw[test_idxs]
    y_train = y_train_full[train_idxs]
    y_test = y_test_full[test_idxs]

    X_test_full = np.vstack((X_test, X_test_2))
    X_train_full = [X_train]
    y_train_full = [y_train]


    for i,(aug_name, mag) in enumerate(cfg.DATASET.AUGMENTATIONS):
        print(f"Applying augmentation: {aug_name}, {mag}")
        if aug_name  == "Random":
            aug_transformer = random_aug.randaugment(cfg.DATASET.RAND_AUGMENT_N_MAX, mag, seed=cfg.DATASET.RAND_SEED+i, transforms=cfg.DATASET.RAND_AUGMENT_AUGS, replace=cfg.DATASET.RAND_AUGMENT_REPLACE, random_n=cfg.DATASET.RAND_AUGMENT_RAND_N)
        else:
            aug_transformer = augmentation_transforms.NAME_TO_TRANSFORM[aug_name].pil_transformer(1.0, mag, X_train[0].shape)
        X_train_aug  = utils.apply_transform(aug_transformer, X_train)
        X_train_full.append(X_train_aug)
        y_train_full.append(y_train)

    X_train_full = np.vstack(X_train_full)
    y_train_full = np.hstack(y_train_full)

    X_train_full_no_corners = X_train_full
    if cfg.DATASET.CORNERS:
      print("Extracting corners...")
      X_train_full = utils.corners_full(X_train_full)
      X_test = utils.corners_full(X_test)

    if cfg.DATASET.ZCA:
        print("Performing ZCA...")
        if cfg.DATASET.ZCA_FULL:
            (X_train_pp, X_test_pp), global_ZCA =  utils.preprocess(np.vstack((X_train_full, X_train_raw[N_train:])),
                                                        X_test, zca_bias=cfg.DATASET.ZCA_BIAS, return_weights=True)
            X_train_pp = X_train_pp[:N_train]
        elif cfg.DATASET.ZCA_EXTRA_AUGMENT:
            _X_train_full = [X_train_full]
            for i in range(cfg.DATASET.NUM_ZCA_EXTRA_AUGMENT):
                aug_transformer = random_aug.randaugment(cfg.DATASET.RAND_AUGMENT_N_MAX, 4, seed=cfg.DATASET.RAND_SEED+i, transforms=cfg.DATASET.RAND_AUGMENT_AUGS, replace=cfg.DATASET.RAND_AUGMENT_REPLACE, random_n=cfg.DATASET.RAND_AUGMENT_RAND_N)
                X_train_aug  = utils.apply_transform(aug_transformer, X_train_full_no_corners)
                if cfg.DATASET.CORNERS:
                    print(f"Extracting corners for extra augment index: {i}")
                    X_train_aug = utils.corners_full(X_train_aug)
                _X_train_full.append(X_train_aug)
            _X_train_full = np.vstack(_X_train_full)
            (X_train_pp, X_test_pp_full), global_ZCA =  utils.preprocess(_X_train_full, X_test_full, zca_bias=cfg.DATASET.ZCA_BIAS, return_weights=True)
            X_train_pp = X_train_pp[:X_train_full.shape[0]]
        else:
            (X_train_pp, X_test_pp_full), global_ZCA =  utils.preprocess(X_train_full, X_test_full, zca_bias=cfg.DATASET.ZCA_BIAS, return_weights=True)
        X_test_pp = X_test_pp_full[:X_test.shape[0]]
        X_test_2_pp = X_test_pp_full[X_test.shape[0]:]
    elif cfg.DATASET.STANDARD_PREPROCESS:
        print("standard preprocess")
        X_train_pp, X_test_pp =  (X_train/255.0).astype('float32'), (X_test/255.0).astype('float32')
        X_test_2_pp = (X_test_2/255.0).astype('float32')
        X_train_pp = (X_train_pp - CIFAR_MEAN[np.newaxis, np.newaxis, np.newaxis, :])/CIFAR_STD[np.newaxis, np.newaxis, np.newaxis, :].astype('float32')
        X_test_pp = (X_test_pp - CIFAR_MEAN[np.newaxis, np.newaxis, np.newaxis, :])/CIFAR_STD[np.newaxis, np.newaxis, np.newaxis, :].astype('float32')
        X_test_2_pp = (X_test_2_pp - CIFAR_MEAN[np.newaxis, np.newaxis, np.newaxis, :])/CIFAR_STD[np.newaxis, np.newaxis, np.newaxis, :].astype('float32')
        global_ZCA = 0
    else:
        X_train_pp, X_test_pp =  (X_train/255.0).astype('float32'), (X_test/255.0).astype('float32')
        X_test_2_pp = (X_test_2/255.0).astype('float32')
        global_ZCA = 0

    np.savez("cifar_10_train.npz", X_train=X_train_pp, y_train=y_train, global_ZCA=global_ZCA)
    np.savez("cifar_10_test.npz", X_test=X_test_pp, y_test=y_test, global_ZCA=global_ZCA)
    np.savez("cifar_10.1.npz", X_test=X_test_2_pp, y_test=y_test_2_full, global_ZCA=global_ZCA)
    meta_data = {"classes" : 10, "d": 1024, "height": 32, "width": 32, "channels": 3, "cfg": cfg_dict}

if __name__ == "__main__":
    preprocess()
