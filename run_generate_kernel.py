import multiprocessing
try:
    multiprocessing.set_start_method("spawn")
except:
    print("Context already set..")

import argparse
import io
import click
import dill
import json
import pickle
import sys
from datetime import datetime
from timeit import default_timer as timer
import concurrent.futures as fs
import copy

import batch_utils
import coatesng
import numpy as np
import scipy.linalg
from sklearn import metrics
from functools import partial
import torch
import boto3
import kernel_gen
import model_repository
import run_train_eval_exp
import utils
import time
from config import arch_short_name, get_cfg_defaults, hash_arch
from ls import eval_ls_model, train_ls_dual_model, train_ls_dual_model_center

JOB_DEF_NAME = "kernel"
JOB_QUEUE_NAME = "kernel_queue2"
BUCKET = "knets"
PREFIX = "kernel_batch_jobs"
CIFAR_10_1_SIZE = 2000

@click.group("cli")
def cli():
    pass

@cli.command('generate_kernel')
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/named_configs/myrtle11_cifar_1024_all_exp.yaml")
def generate_kernel(config_updates,  config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)
    cfg_dict = utils.cfg_to_dict(cfg)
    if cfg.SYSTEM.NUM_GPUS == 0:
        cfg.SYSTEM.NUM_GPUS = torch.cuda.device_count()
    cfg.freeze()

    m_repo = model_repository.ModelRepository()
    dataset_a = cfg.DATASET.TRAIN_NAME
    dataset_b = cfg.DATASET.TEST_NAME
    # load dataset
    print(f"Generating kernel between {dataset_a} and {dataset_b}")
    generate_kernel_only(cfg, dataset_a, dataset_b)

@cli.command('solve_kernels')
@click.argument("train_checkpoint_id")
@click.argument("test_checkpoint_id")
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/named_configs/myrtle11_cifar_1024_all_exp.yaml")
def solve_kernels(train_checkpoint_id, test_checkpoint_id, config_updates, config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)
    cfg_dict = utils.cfg_to_dict(cfg)
    if cfg.SYSTEM.NUM_GPUS == 0:
        cfg.SYSTEM.NUM_GPUS = torch.cuda.device_count()
    cfg.freeze()
    return _solve_kernels(cfg, train_checkpoint_id, test_checkpoint_id)

def _solve_kernels(cfg, train_checkpoint_id, test_checkpoint_id):
    m_repo = model_repository.ModelRepository()
    K_train = utils.bytes_to_numpy(m_repo.get_checkpoint_data(train_checkpoint_id, data_type="kernel"))
    K_test = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test_checkpoint_id, data_type="kernel")).T
    train_checkpoint = m_repo.get_checkpoint(train_checkpoint_id)
    test_checkpoint = m_repo.get_checkpoint(test_checkpoint_id)

    print(train_checkpoint.extra_info["x_dataset"], train_checkpoint.extra_info["y_dataset"])
    assert train_checkpoint.extra_info["x_dataset"] == train_checkpoint.extra_info["y_dataset"]
    assert test_checkpoint.extra_info["x_dataset"] == train_checkpoint.extra_info["x_dataset"]

    train_dataset = train_checkpoint.extra_info["x_dataset"]
    test_dataset = test_checkpoint.extra_info["y_dataset"]


    train_dataset_obj = m_repo.get_dataset_by_name(train_dataset)
    test_dataset_obj = m_repo.get_dataset_by_name(test_dataset)

    train_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(train_dataset_obj.uuid))))
    test_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(test_dataset_obj.uuid))))

    if "y_test" in test_dataset_npz.keys():
        y_test = test_dataset_npz["y_test"][:cfg.DATASET.TEST_SUBSET]
    else:
        y_test = test_dataset_npz["y_train"][:cfg.DATASET.TEST_SUBSET]
    y_train  = train_dataset_npz["y_train"][:cfg.DATASET.TRAIN_SUBSET]
    print("all close", np.allclose(K_train, K_train.T))
    K_train = K_train[:cfg.DATASET.TRAIN_SUBSET, :cfg.DATASET.TRAIN_SUBSET]
    K_test = K_test[:cfg.DATASET.TEST_SUBSET, :cfg.DATASET.TRAIN_SUBSET]
    print("Max difference across diag", np.max(np.abs(K_train - K_train.T)))
    K_train_inv = np.linalg.inv(K_train)
    y_train_enc = np.eye(10)[y_train]
    y_train_loo = np.argmax(y_train_enc - K_train_inv.dot(y_train_enc)/np.diag(K_train_inv)[:, np.newaxis], axis=1)
    print("loo accuracy:")
    print(metrics.accuracy_score(y_train_loo, y_train))

    run_train_eval_exp.solve_kernel(cfg, K_train, y_train, K_test, y_test)

@cli.command('glue_solve_cifar_10')
@click.argument("train_kernel_uuid")
@click.argument("test_kernel_uuid")
@click.argument("test2_kernel_uuid")
@click.argument("test_test_kernel_uuid")
@click.argument("test2_test2_kernel_uuid")
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/named_configs/myrtle11_cifar_1024_all_exp.yaml")
def glue_solve_cifar_10(train_kernel_uuid, test_kernel_uuid, test2_kernel_uuid, test_test_kernel_uuid, test2_test2_kernel_uuid, config_updates, config):
    _glue_solve_cifar_10(train_kernel_uuid, test_kernel_uuid, test2_kernel_uuid, test_test_kernel_uuid, test2_test2_kernel_uuid, config_updates, config)



def _glue_solve_cifar_10(train_kernel_uuid, test_kernel_uuid, test2_kernel_uuid, test_test_kernel_uuid, test2_test2_kernel_uuid, config_updates, config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)

    m_repo = model_repository.ModelRepository()
    dataset_id = cfg.DATASET.TEST_NAME.split("-")[-1]
    assert dataset_id == cfg.DATASET.TRAIN_NAME.split("-")[-1]
    cifar_10_1_test_name = f"cifar-10.1-test-{dataset_id}"

    train_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TRAIN_NAME)
    test_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TEST_NAME)
    test2_dataset_obj = m_repo.get_dataset_by_name(cifar_10_1_test_name)

    train_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(train_dataset_obj.uuid))))
    test_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(test_dataset_obj.uuid))))
    test2_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(test2_dataset_obj.uuid))))

    y_test = test_dataset_npz["y_test"][:cfg.DATASET.TEST_SUBSET]
    y_test2 = test2_dataset_npz["y_test"][:CIFAR_10_1_SIZE]
    y_train = train_dataset_npz["y_train"][:cfg.DATASET.TRAIN_SUBSET]
    print("gluing train kernel...")
    cfg_train_kernel_glue = copy.deepcopy(cfg)
    cfg_train_kernel_glue.DATASET.KERNEL_UUID = train_kernel_uuid
    train_kernel_cpt_id = _glue_kernels(cfg_train_kernel_glue)
    print(f"glued train kernel checkpoint_id: {train_kernel_cpt_id}...")

    print("gluing test kernel...")
    cfg_test_kernel_glue = copy.deepcopy(cfg)
    cfg_test_kernel_glue.DATASET.KERNEL_UUID = test_kernel_uuid
    test_kernel_cpt_id = _glue_kernels(cfg_test_kernel_glue)
    print(f"glued test kernel checkpoint_id: {test_kernel_cpt_id}...")


    print("gluing test2 kernel...")
    cfg_test2_kernel_glue = copy.deepcopy(cfg)
    cfg_test2_kernel_glue.DATASET.KERNEL_UUID = test2_kernel_uuid
    test2_kernel_cpt_id = _glue_kernels(cfg_test2_kernel_glue)
    print(f"glued test2 kernel checkpoint_id: {test2_kernel_cpt_id}...")

    print("gluing test_test kernel...")
    cfg_test_test_kernel_glue = copy.deepcopy(cfg)
    cfg_test_test_kernel_glue.DATASET.KERNEL_UUID = test_test_kernel_uuid
    test_test_kernel_cpt_id = _glue_kernels(cfg_test_test_kernel_glue)
    print(f"glued test_test kernel checkpoint_id: {test_test_kernel_cpt_id}...")

    print("gluing test2_test2 kernel...")
    cfg_test2_test2_kernel_glue = copy.deepcopy(cfg)
    cfg_test2_test2_kernel_glue.DATASET.KERNEL_UUID = test2_test2_kernel_uuid
    test2_test2_kernel_cpt_id = _glue_kernels(cfg_test2_test2_kernel_glue)
    print(f"glued test2_test2 kernel checkpoint_id: {test2_test2_kernel_cpt_id}...")

    print(f"Kernel checkpoint ids (train, test, test2 test_test, test2_test2): {train_kernel_cpt_id, test_kernel_cpt_id, test2_kernel_cpt_id, test_test_kernel_cpt_id, test2_test2_kernel_cpt_id}")

    K_train = utils.bytes_to_numpy(m_repo.get_checkpoint_data(train_kernel_cpt_id, data_type="kernel"))
    K_test = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test_kernel_cpt_id, data_type="kernel")).T
    K_test2 = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test2_kernel_cpt_id, data_type="kernel")).T
    K_test_test = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test_test_kernel_cpt_id, data_type="kernel"))
    K_test2_test2 = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test2_test2_kernel_cpt_id, data_type="kernel"))

    K_train = K_train[:cfg.DATASET.TRAIN_SUBSET, :cfg.DATASET.TRAIN_SUBSET]
    K_test = K_test[:cfg.DATASET.TEST_SUBSET, :cfg.DATASET.TRAIN_SUBSET]
    K_test_test = K_test_test[:cfg.DATASET.TEST_SUBSET, :cfg.DATASET.TEST_SUBSET]
    K_test2 = K_test2[:, :cfg.DATASET.TRAIN_SUBSET]


    LOO_TILT_FACTOR = 0.3
    # normalize the kernel
    test_norms = np.sqrt(np.diag(K_test_test))[:, np.newaxis]
    test2_norms = np.sqrt(np.diag(K_test2_test2))[:, np.newaxis]
    train_norms = np.sqrt(np.diag(K_train))[:, np.newaxis]
    K_train = (K_train / train_norms)/train_norms.T
    K_test = (K_test / train_norms.T)/test_norms
    K_test2 = (K_test2 / train_norms.T)/test2_norms

    K_train_inv = np.linalg.inv(K_train)
    y_train_enc = np.eye(10)[y_train]
    alpha_0 = K_train_inv.dot(y_train_enc)
    y_train_loo_logits = y_train_enc - K_train_inv.dot(y_train_enc)/np.diag(K_train_inv)[:, np.newaxis]
    y_train_loo = np.argmax(y_train_loo_logits, axis=1)

    test_acc  = metrics.accuracy_score(np.argmax(K_test.dot(alpha_0), axis=1), y_test)
    test2_acc = metrics.accuracy_score(np.argmax(K_test2.dot(alpha_0), axis=1), y_test2)


    loo_accs = []
    for loo_tilt_factor in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        alphas_loo_tilt = K_train_inv.dot(y_train_enc - loo_tilt_factor*y_train_loo_logits)
        test_acc_with_loo = metrics.accuracy_score(np.argmax(K_test.dot(alphas_loo_tilt), axis=1), y_test)
        test2_acc_with_loo = metrics.accuracy_score(np.argmax(K_test2.dot(alphas_loo_tilt), axis=1), y_test2)
        loo_accs.append((test_acc_with_loo, test2_acc_with_loo, loo_tilt_factor, alphas_loo_tilt))


    test_acc_with_loo, test2_acc_with_loo, loo_tilt_factor, alphas_loo_tilt = max(loo_accs, key=lambda x: x[0])

    train_checkpoint = m_repo.get_checkpoint(uuid=train_kernel_cpt_id)
    test_logits = K_test.dot(alphas_loo_tilt)
    test2_logits = K_test2.dot(alphas_loo_tilt)

    test_result = {}
    test_result["kernel"] = K_test
    test_result["preds"] = np.argmax(test_logits, axis=1)
    test_result["logits"] = test_logits
    test_result["acc_with_loo"] = test_acc_with_loo
    test_result["acc_without_loo"] = test_acc
    test_result["loo_tilt_factor"] = loo_tilt_factor
    test_result["alpha"] = alphas_loo_tilt
    test_result["alpha_no_tilt"] = alpha_0

    test_eval = m_repo.create_evaluation(checkpoint_uuid=train_checkpoint.uuid, evaluation_set_uuid=test_dataset_obj.uuid, evaluation_set_size=test_logits.shape[0], evaluation_set_indices_bytes=utils.numpy_to_bytes(np.arange(test_logits.shape[0])), metric="accuracy", value=test_acc_with_loo, predictions_data_bytes=dill.dumps(test_result, protocol=4), extra_info={})

    test2_result = {}
    test2_result["kernel"] = K_test2
    test2_result["preds"] = np.argmax(test2_logits, axis=1)
    test2_result["logits"] = test2_logits
    test2_result["acc_with_loo"] = test2_acc_with_loo
    test2_result["acc_without_loo"] = test2_acc
    test2_result["loo_tilt_factor"] = loo_tilt_factor
    test2_result["alpha"] = alphas_loo_tilt
    test2_result["alpha_no_tilt"] = alpha_0

    test2_eval = m_repo.create_evaluation(checkpoint_uuid=train_checkpoint.uuid, evaluation_set_uuid=test2_dataset_obj.uuid, evaluation_set_size=test2_logits.shape[0], evaluation_set_indices_bytes=utils.numpy_to_bytes(np.arange(CIFAR_10_1_SIZE)), metric="accuracy", value=test2_acc_with_loo, predictions_data_bytes=dill.dumps(test2_result, protocol=4), extra_info={})


    print("cifar10 LOO accuracy:", metrics.accuracy_score(y_train_loo, y_train))
    print("cifar10 test accuracy ", test_acc)
    print("cifar10.1 test accuracy ", test2_acc)
    print("cifar10 test accuracy (with loo-tilt)", test_acc_with_loo)
    print("cifar10.1 accuracy (with loo-tilt)", test2_acc_with_loo)
    print(f"Kernel checkpoint ids (train, test, test2 test_test, test2_test2): {train_kernel_cpt_id, test_kernel_cpt_id, test2_kernel_cpt_id, test_test_kernel_cpt_id, test2_test2_kernel_cpt_id}")

def _glue_solve_dataset(train_kernel_uuid, test_kernel_uuid, test_test_kernel_uuid, config_updates, config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)

    m_repo = model_repository.ModelRepository()
    dataset_id = cfg.DATASET.TEST_NAME.split("-")[-1]
    assert dataset_id == cfg.DATASET.TRAIN_NAME.split("-")[-1]

    train_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TRAIN_NAME)
    test_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TEST_NAME)

    train_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(train_dataset_obj.uuid))))
    test_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(test_dataset_obj.uuid))))

    y_test = test_dataset_npz["y_test"][:cfg.DATASET.TEST_SUBSET]
    y_train = train_dataset_npz["y_train"][:cfg.DATASET.TRAIN_SUBSET]
    print("gluing train kernel...")
    cfg_train_kernel_glue = copy.deepcopy(cfg)
    cfg_train_kernel_glue.DATASET.KERNEL_UUID = train_kernel_uuid
    train_kernel_cpt_id = _glue_kernels(cfg_train_kernel_glue)
    print(f"glued train kernel checkpoint_id: {train_kernel_cpt_id}...")

    print("gluing test kernel...")
    cfg_test_kernel_glue = copy.deepcopy(cfg)
    cfg_test_kernel_glue.DATASET.KERNEL_UUID = test_kernel_uuid
    test_kernel_cpt_id = _glue_kernels(cfg_test_kernel_glue)
    print(f"glued test kernel checkpoint_id: {test_kernel_cpt_id}...")



    print("gluing test_test kernel...")
    cfg_test_test_kernel_glue = copy.deepcopy(cfg)
    cfg_test_test_kernel_glue.DATASET.KERNEL_UUID = test_test_kernel_uuid
    test_test_kernel_cpt_id = _glue_kernels(cfg_test_test_kernel_glue)
    print(f"glued test_test kernel checkpoint_id: {test_test_kernel_cpt_id}...")



    K_train = utils.bytes_to_numpy(m_repo.get_checkpoint_data(train_kernel_cpt_id, data_type="kernel"))
    K_test = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test_kernel_cpt_id, data_type="kernel")).T
    K_test_test = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test_test_kernel_cpt_id, data_type="kernel"))

    K_train = K_train[:cfg.DATASET.TRAIN_SUBSET, :cfg.DATASET.TRAIN_SUBSET]
    K_test = K_test[:cfg.DATASET.TEST_SUBSET, :cfg.DATASET.TRAIN_SUBSET]
    K_test_test = K_test_test[:cfg.DATASET.TEST_SUBSET, :cfg.DATASET.TEST_SUBSET]


    LOO_TILT_FACTOR = 0.3
    # normalize the kernel
    test_norms = np.sqrt(np.diag(K_test_test))[:, np.newaxis]
    train_norms = np.sqrt(np.diag(K_train))[:, np.newaxis]
    K_train = (K_train / train_norms)/train_norms.T
    K_test = (K_test / train_norms.T)/test_norms

    K_train_inv = np.linalg.inv(K_train)
    y_train_enc = np.eye(10)[y_train]
    alpha_0 = K_train_inv.dot(y_train_enc)
    y_train_loo_logits = y_train_enc - K_train_inv.dot(y_train_enc)/np.diag(K_train_inv)[:, np.newaxis]
    y_train_loo = np.argmax(y_train_loo_logits, axis=1)

    test_acc  = metrics.accuracy_score(np.argmax(K_test.dot(alpha_0), axis=1), y_test)


    loo_accs = []
    for loo_tilt_factor in range(100):
        loo_tilt_factor = loo_tilt_factor/100
        alphas_loo_tilt = K_train_inv.dot(y_train_enc - loo_tilt_factor*y_train_loo_logits)
        test_acc_with_loo = metrics.accuracy_score(np.argmax(K_test.dot(alphas_loo_tilt), axis=1), y_test)
        loo_accs.append((test_acc_with_loo, loo_tilt_factor, alphas_loo_tilt))


    test_acc_with_loo, loo_tilt_factor, alphas_loo_tilt = max(loo_accs, key=lambda x: x[0])

    train_checkpoint = m_repo.get_checkpoint(uuid=train_kernel_cpt_id)
    test_logits = K_test.dot(alphas_loo_tilt)

    test_result = {}
    test_result["kernel"] = K_test
    test_result["preds"] = np.argmax(test_logits, axis=1)
    test_result["logits"] = test_logits
    test_result["acc_with_loo"] = test_acc_with_loo
    test_result["acc_without_loo"] = test_acc
    test_result["loo_tilt_factor"] = loo_tilt_factor
    test_result["alpha"] = alphas_loo_tilt
    test_result["alpha_no_tilt"] = alpha_0

    test_eval = m_repo.create_evaluation(checkpoint_uuid=train_checkpoint.uuid, evaluation_set_uuid=test_dataset_obj.uuid, evaluation_set_size=test_logits.shape[0], evaluation_set_indices_bytes=utils.numpy_to_bytes(np.arange(test_logits.shape[0])), metric="accuracy", value=test_acc_with_loo, predictions_data_bytes=dill.dumps(test_result, protocol=4), extra_info={})
    print("dataset LOO accuracy:", metrics.accuracy_score(y_train_loo, y_train))
    print("dataset test accuracy ", test_acc)
    print("dataset test accuracy (with loo-tilt)", test_acc_with_loo)
    print(f"Kernel checkpoint ids (train, test, test_test): {train_kernel_cpt_id, test_kernel_cpt_id, test_test_kernel_cpt_id}")

@cli.command('kernel_solve_cifar_10')
@click.argument("train_kernel_cpt_id")
@click.argument("test_kernel_cpt_id")
@click.argument("test2_kernel_cpt_id")
@click.argument("test_test_kernel_cpt_id")
@click.argument("test2_test2_kernel_cpt_id")
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/named_configs/myrtle11_cifar_1024_all_exp.yaml")
def kernel_solve_cifar_10(train_kernel_cpt_id, test_kernel_cpt_id, test2_kernel_cpt_id, test_test_kernel_cpt_id, test2_test2_kernel_cpt_id, config_updates, config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)

    m_repo = model_repository.ModelRepository()
    m_repo = model_repository.ModelRepository()
    dataset_id = cfg.DATASET.TEST_NAME.split("-")[-1]
    assert dataset_id == cfg.DATASET.TRAIN_NAME.split("-")[-1]
    cifar_10_1_test_name = f"cifar-10.1-test-{dataset_id}"

    train_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TRAIN_NAME)
    test_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TEST_NAME)
    test2_dataset_obj = m_repo.get_dataset_by_name(cifar_10_1_test_name)

    train_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(train_dataset_obj.uuid))))
    test_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(test_dataset_obj.uuid))))
    test2_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(test2_dataset_obj.uuid))))

    y_test = test_dataset_npz["y_test"][:cfg.DATASET.TEST_SUBSET]
    y_test2 = test2_dataset_npz["y_test"][:CIFAR_10_1_SIZE]
    y_train = train_dataset_npz["y_train"][:cfg.DATASET.TRAIN_SUBSET]

    print("Downloading kernels...")

    K_train = utils.bytes_to_numpy(m_repo.get_checkpoint_data(train_kernel_cpt_id, data_type="kernel"))
    K_test = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test_kernel_cpt_id, data_type="kernel")).T
    K_test2 = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test2_kernel_cpt_id, data_type="kernel")).T
    K_test_test = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test_test_kernel_cpt_id, data_type="kernel"))
    K_test2_test2 = utils.bytes_to_numpy(m_repo.get_checkpoint_data(test2_test2_kernel_cpt_id, data_type="kernel"))

    K_train = K_train[:cfg.DATASET.TRAIN_SUBSET, :cfg.DATASET.TRAIN_SUBSET]
    K_test = K_test[:cfg.DATASET.TEST_SUBSET, :cfg.DATASET.TRAIN_SUBSET]
    K_test_test = K_test_test[:cfg.DATASET.TEST_SUBSET, :cfg.DATASET.TEST_SUBSET]
    K_test2 = K_test2[:, :cfg.DATASET.TRAIN_SUBSET]


    LOO_TILT_FACTOR = 0.3
    # normalize the kernel
    test_norms = np.sqrt(np.diag(K_test_test))[:, np.newaxis]
    test2_norms = np.sqrt(np.diag(K_test2_test2))[:, np.newaxis]
    train_norms = np.sqrt(np.diag(K_train))[:, np.newaxis]
    K_train = (K_train / train_norms)/train_norms.T
    K_test = (K_test / train_norms.T)/test_norms
    K_test2 = (K_test2 / train_norms.T)/test2_norms

    K_train_inv = np.linalg.inv(K_train)
    y_train_enc = np.eye(10)[y_train]
    alpha_0 = K_train_inv.dot(y_train_enc)
    y_train_loo_logits = y_train_enc - K_train_inv.dot(y_train_enc)/np.diag(K_train_inv)[:, np.newaxis]
    y_train_loo = np.argmax(y_train_loo_logits, axis=1)

    test_acc  = metrics.accuracy_score(np.argmax(K_test.dot(alpha_0), axis=1), y_test)
    test2_acc = metrics.accuracy_score(np.argmax(K_test2.dot(alpha_0), axis=1), y_test2)


    loo_accs = []
    for loo_tilt_factor in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        alphas_loo_tilt = K_train_inv.dot(y_train_enc - loo_tilt_factor*y_train_loo_logits)
        test_acc_with_loo = metrics.accuracy_score(np.argmax(K_test.dot(alphas_loo_tilt), axis=1), y_test)
        test2_acc_with_loo = metrics.accuracy_score(np.argmax(K_test2.dot(alphas_loo_tilt), axis=1), y_test2)
        loo_accs.append((test_acc_with_loo, test2_acc_with_loo, loo_tilt_factor, alphas_loo_tilt))


    test_acc_with_loo, test2_acc_with_loo, loo_tilt_factor, alphas_loo_tilt = max(loo_accs, key=lambda x: x[0])

    train_checkpoint = m_repo.get_checkpoint(uuid=train_kernel_cpt_id)
    test_logits = K_test.dot(alphas_loo_tilt)
    test2_logits = K_test2.dot(alphas_loo_tilt)

    test_result = {}
    test_result["kernel"] = K_test
    test_result["preds"] = np.argmax(test_logits, axis=1)
    test_result["logits"] = test_logits
    test_result["acc_with_loo"] = test_acc_with_loo
    test_result["acc_without_loo"] = test_acc
    test_result["loo_tilt_factor"] = loo_tilt_factor
    test_result["alpha"] = alphas_loo_tilt
    test_result["alpha_no_tilt"] = alpha_0

    test_eval = m_repo.create_evaluation(checkpoint_uuid=train_checkpoint.uuid, evaluation_set_uuid=test_dataset_obj.uuid, evaluation_set_size=test_logits.shape[0], evaluation_set_indices_bytes=utils.numpy_to_bytes(np.arange(test_logits.shape[0])), metric="accuracy", value=test_acc_with_loo, predictions_data_bytes=dill.dumps(test_result, protocol=4), extra_info={})

    test2_result = {}
    test2_result["kernel"] = K_test2
    test2_result["preds"] = np.argmax(test2_logits, axis=1)
    test2_result["logits"] = test2_logits
    test2_result["acc_with_loo"] = test2_acc_with_loo
    test2_result["acc_without_loo"] = test2_acc
    test2_result["loo_tilt_factor"] = loo_tilt_factor
    test2_result["alpha"] = alphas_loo_tilt
    test2_result["alpha_no_tilt"] = alpha_0

    test2_eval = m_repo.create_evaluation(checkpoint_uuid=train_checkpoint.uuid, evaluation_set_uuid=test2_dataset_obj.uuid, evaluation_set_size=test2_logits.shape[0], evaluation_set_indices_bytes=utils.numpy_to_bytes(np.arange(CIFAR_10_1_SIZE)), metric="accuracy", value=test2_acc_with_loo, predictions_data_bytes=dill.dumps(test2_result, protocol=4), extra_info={})


    print("cifar10 LOO accuracy:", metrics.accuracy_score(y_train_loo, y_train))
    print("cifar10 test accuracy ", test_acc)
    print("cifar10.1 test accuracy ", test2_acc)
    print("cifar10 test accuracy (with loo-tilt)", test_acc_with_loo)
    print("cifar10.1 accuracy (with loo-tilt)", test2_acc_with_loo)
    print(f"Kernel checkpoint ids (train, test, test2 test_test, test2_test2): {train_kernel_cpt_id, test_kernel_cpt_id, test2_kernel_cpt_id, test_test_kernel_cpt_id, test2_test2_kernel_cpt_id}")

@cli.command('kernel_generate_solve_cifar_10')
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/named_configs/myrtle11_cifar_1024_all_exp.yaml")
def kernel_generate_solve_cifar_10(config_updates, config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)
    if cfg.SYSTEM.NUM_GPUS == 0:
        cfg.SYSTEM.NUM_GPUS = torch.cuda.device_count()
    m_repo = model_repository.ModelRepository()
    # load dataset
    dataset_a = cfg.DATASET.TRAIN_NAME
    if cfg.SYSTEM.USE_AWS_BATCH and cfg.SYSTEM.AWS_BATCH_CORES > 0:
        batch_utils._update_compute_env("kernels3", cfg.SYSTEM.AWS_BATCH_CORES)
    print("generating sharded train kernel...")

    # make all 5 configs
    # 1. Train x Train Kernel
    # 2. Train x Test Kernel
    # 3. Test x Test Kernel
    # 4. Train x Test2 Kernel
    # 5. Test2 x Test2 Kernel
    if cfg.KERNEL.COATESNG.ON:
        cn_cfg = cfg.KERNEL.COATESNG
        train_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TRAIN_NAME)
        train_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(train_dataset_obj.uuid))))
        X_train = train_npz["X_train"]
        net = coatesng.build_featurizer(patch_size=cn_cfg.PATCH_SIZE, pool_size=cn_cfg.POOL_SIZE, pool_stride=cn_cfg.POOL_STRIDE, bias=cn_cfg.BIAS, X_train=X_train, filter_batch_size=cn_cfg.FILTER_BATCH_SIZE, num_filters=cn_cfg.NUM_FILTERS, filter_scale=cn_cfg.FILTER_SCALE, num_channels=3, patch_distribution=cn_cfg.PATCH_DISTRIBUTION, seed=cn_cfg.SEED, normalize_patches=cn_cfg.NORMALIZE_PATCHES)
        with open("/dev/shm/featurizer.pickle", "wb") as f:
            f.write(pickle.dumps(net))

    cfg_test_kernel = copy.deepcopy(cfg)

    cfg_train_kernel = copy.deepcopy(cfg)
    cfg_train_kernel.DATASET.TEST_NAME = cfg.DATASET.TRAIN_NAME
    cfg_train_kernel.DATASET.TEST_SUBSET = cfg.DATASET.TRAIN_SUBSET

    cfg_test_test_kernel = copy.deepcopy(cfg)
    cfg_test_test_kernel.DATASET.TRAIN_NAME = cfg.DATASET.TEST_NAME
    cfg_test_test_kernel.DATASET.TRAIN_SUBSET = cfg.DATASET.TEST_SUBSET

    dataset_id = cfg.DATASET.TEST_NAME.split("-")[-1]
    assert dataset_id == cfg.DATASET.TRAIN_NAME.split("-")[-1]
    cifar_10_1_test_name = f"cifar-10.1-test-{dataset_id}"
    cfg_test2_kernel = copy.deepcopy(cfg)
    cfg_test2_kernel.DATASET.TEST_NAME = cifar_10_1_test_name
    cfg_test2_kernel.DATASET.TEST_SUBSET = CIFAR_10_1_SIZE

    cfg_test2_test2_kernel = copy.deepcopy(cfg)
    cfg_test2_test2_kernel.DATASET.TRAIN_NAME = cifar_10_1_test_name
    cfg_test2_test2_kernel.DATASET.TEST_NAME = cifar_10_1_test_name
    cfg_test2_test2_kernel.DATASET.TRAIN_SUBSET = CIFAR_10_1_SIZE
    cfg_test2_test2_kernel.DATASET.TEST_SUBSET = CIFAR_10_1_SIZE

    train_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TRAIN_NAME)
    test_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TEST_NAME)
    test2_dataset_obj = m_repo.get_dataset_by_name(cifar_10_1_test_name)

    train_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(train_dataset_obj.uuid))))
    test_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(test_dataset_obj.uuid))))
    test2_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(test2_dataset_obj.uuid))))

    y_test = test_dataset_npz["y_test"][:cfg.DATASET.TEST_SUBSET]
    y_test2 = test2_dataset_npz["y_test"][:CIFAR_10_1_SIZE]
    y_train = train_dataset_npz["y_train"][:cfg.DATASET.TRAIN_SUBSET]
    print("y_test2", y_test2)
    print("y_test", y_test)

    if cfg.SYSTEM.USE_AWS_BATCH:
        executor = fs.ThreadPoolExecutor(5)
        print("generating sharded train kernel...")
        train_kernel_uuid_future = executor.submit(_generate_kernel_all_parts, cfg_train_kernel)

        print("generating sharded test kernel...")
        test_kernel_uuid_future = executor.submit(_generate_kernel_all_parts, cfg_test_kernel)

        print("generating sharded test_test kernel...")
        test_test_kernel_uuid_future = executor.submit(_generate_kernel_all_parts, cfg_test_test_kernel)

        print("generating sharded test2 kernel...")
        test2_kernel_uuid_future = executor.submit(_generate_kernel_all_parts, cfg_test2_kernel)

        print("generating sharded test2_test2 kernel...")
        test2_test2_kernel_uuid_future = executor.submit(_generate_kernel_all_parts, cfg_test2_test2_kernel)

        fs.wait([test_kernel_uuid_future, train_kernel_uuid_future])
        train_kernel_uuid = train_kernel_uuid_future.result()
        print(f"generated sharded train kernel uuid: {train_kernel_uuid}...")
        test_kernel_uuid = test_kernel_uuid_future.result()
        print(f"generated sharded test kernel uuid: {test_kernel_uuid}...")

        test_test_kernel_uuid = test_test_kernel_uuid_future.result()
        print(f"generated sharded test_test kernel uuid: {test_test_kernel_uuid}...")

        test2_kernel_uuid = test2_kernel_uuid_future.result()
        print(f"generated sharded test2 kernel uuid: {test2_kernel_uuid}...")
        test2_test2_kernel_uuid = test2_test2_kernel_uuid_future.result()
        print(f"generated sharded test2_test2 kernel uuid: {test2_test2_kernel_uuid}...")
    else:
        print("generating sharded test2 kernel...")
        test2_kernel_uuid = _generate_kernel_all_parts(cfg_test2_kernel)
        print("generating sharded train kernel...")
        train_kernel_uuid = _generate_kernel_all_parts(cfg_train_kernel)
        print("generating sharded test kernel...")
        test_kernel_uuid = _generate_kernel_all_parts(cfg_test_kernel)
        print("generating sharded test_test kernel...")
        test_test_kernel_uuid = _generate_kernel_all_parts(cfg_test_test_kernel)
        print("generating sharded test2_test2 kernel...")
        test2_test2_kernel_uuid = _generate_kernel_all_parts(cfg_test2_test2_kernel)
    print("====KERNEL GENERATION DONE========")
    print("=================================\n"*5)
    print(train_kernel_uuid, test_kernel_uuid, test2_kernel_uuid, test_test_kernel_uuid, test2_test2_kernel_uuid, config_updates, config)
    print("=================================\n"*5)
    return _glue_solve_cifar_10(train_kernel_uuid, test_kernel_uuid, test2_kernel_uuid, test_test_kernel_uuid, test2_test2_kernel_uuid, config_updates, config)


@cli.command('kernel_generate_solve_dataset')
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/named_configs/myrtle11_cifar_1024_all_exp.yaml")
def kernel_generate_solve_dataset(config_updates, config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)
    if cfg.SYSTEM.NUM_GPUS == 0:
        cfg.SYSTEM.NUM_GPUS = torch.cuda.device_count()
    m_repo = model_repository.ModelRepository()
    # load dataset
    dataset_a = cfg.DATASET.TRAIN_NAME
    if cfg.SYSTEM.USE_AWS_BATCH and cfg.SYSTEM.AWS_BATCH_CORES > 0:
        batch_utils._update_compute_env("kernels3", cfg.SYSTEM.AWS_BATCH_CORES)
    print("generating sharded train kernel...")

    # make all 3 configs
    # 1. Train x Train Kernel
    # 2. Train x Test Kernel
    # 3. Test x Test Kernel

    cfg_test_kernel = copy.deepcopy(cfg)

    cfg_train_kernel = copy.deepcopy(cfg)
    cfg_train_kernel.DATASET.TEST_NAME = cfg.DATASET.TRAIN_NAME
    cfg_train_kernel.DATASET.TEST_SUBSET = cfg.DATASET.TRAIN_SUBSET

    cfg_test_test_kernel = copy.deepcopy(cfg)
    cfg_test_test_kernel.DATASET.TRAIN_NAME = cfg.DATASET.TEST_NAME
    cfg_test_test_kernel.DATASET.TRAIN_SUBSET = cfg.DATASET.TEST_SUBSET

    dataset_id = cfg.DATASET.TEST_NAME.split("-")[-1]
    train_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TRAIN_NAME)
    test_dataset_obj = m_repo.get_dataset_by_name(cfg.DATASET.TEST_NAME)

    train_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(train_dataset_obj.uuid))))
    test_dataset_npz = np.load(io.BytesIO(m_repo.get_dataset_data(str(test_dataset_obj.uuid))))

    y_test = test_dataset_npz["y_test"][:cfg.DATASET.TEST_SUBSET]
    y_train = train_dataset_npz["y_train"][:cfg.DATASET.TRAIN_SUBSET]
    print("y_test", y_test)

    if cfg.SYSTEM.USE_AWS_BATCH:
        executor = fs.ThreadPoolExecutor(5)
        print("generating sharded train kernel...")
        train_kernel_uuid_future = executor.submit(_generate_kernel_all_parts, cfg_train_kernel)

        print("generating sharded test kernel...")
        test_kernel_uuid_future = executor.submit(_generate_kernel_all_parts, cfg_test_kernel)

        print("generating sharded test_test kernel...")
        test_test_kernel_uuid_future = executor.submit(_generate_kernel_all_parts, cfg_test_test_kernel)

        fs.wait([test_kernel_uuid_future, train_kernel_uuid_future, test_test_kernel_uuid_future])
        train_kernel_uuid = train_kernel_uuid_future.result()
        print(f"generated sharded train kernel uuid: {train_kernel_uuid}...")
        test_kernel_uuid = test_kernel_uuid_future.result()
        print(f"generated sharded test kernel uuid: {test_kernel_uuid}...")
        test_test_kernel_uuid = test_test_kernel_uuid_future.result()
        print(f"generated sharded test_test kernel uuid: {test_test_kernel_uuid}...")

    else:
        print("generating sharded train kernel...")
        train_kernel_uuid = _generate_kernel_all_parts(cfg_train_kernel)
        print("generating sharded test kernel...")
        test_kernel_uuid = _generate_kernel_all_parts(cfg_test_kernel)
        print("generating sharded test_test kernel...")
        test_test_kernel_uuid = _generate_kernel_all_parts(cfg_test_test_kernel)

    print("====KERNEL GENERATION DONE========")
    print("=================================\n"*5)
    print(train_kernel_uuid, test_kernel_uuid, test_test_kernel_uuid,  config_updates, config)
    print("=================================\n"*5)
    return _glue_solve_dataset(train_kernel_uuid, test_kernel_uuid, test_test_kernel_uuid, config_updates, config)


@cli.command('generate_kernel_all_parts')
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/named_configs/myrtle11_cifar_1024_all_exp.yaml")
def generate_kernel_all_parts(config_updates, config):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)
    cfg_dict = utils.cfg_to_dict(cfg)
    if cfg.SYSTEM.NUM_GPUS == 0:
        cfg.SYSTEM.NUM_GPUS = torch.cuda.device_count()
    return _generate_kernel_all_parts(cfg)

def _generate_kernel_all_parts(cfg):
    m_repo = model_repository.ModelRepository()
    # load dataset
    dataset_a = cfg.DATASET.TRAIN_NAME
    dataset_b = cfg.DATASET.TEST_NAME
    use_aws_batch = cfg.SYSTEM.USE_AWS_BATCH
    print(f"Generating kernel between {dataset_a} and {dataset_b}")
    assert cfg.DATASET.SHARD_X_SIZE != 0
    assert cfg.DATASET.SHARD_Y_SIZE != 0
    x_shard_size = int(cfg.DATASET.SHARD_X_SIZE)
    y_shard_size = int(cfg.DATASET.SHARD_Y_SIZE)
    dataset_A = m_repo.get_dataset_by_name(dataset_a)
    print("dataset_b", dataset_b)
    dataset_B = m_repo.get_dataset_by_name(dataset_b)
    x_size = cfg.DATASET.TRAIN_SUBSET
    y_size = cfg.DATASET.TEST_SUBSET
    print("x size", x_size)
    print("y size", y_size)
    m_repo = model_repository.ModelRepository()
    kernel_uuid = m_repo.gen_short_uuid()
    symmetric = dataset_a == dataset_b
    batch_job_ids = {}
    for i,start_x in enumerate(range(0, x_size, x_shard_size)):
        for j, start_y in enumerate(range(0, y_size, y_shard_size)):
            if symmetric and i > j:
                print(f"skipping kernel indices {i}, {j} becauase matrix is symmetric")
                continue
            cfg_part = copy.deepcopy(cfg)
            cfg_part.SYSTEM.USE_AWS_BATCH = False
            cfg_part.DATASET.KERNEL_UUID = kernel_uuid
            cfg_part.DATASET.SHARD_X_IDX = i
            cfg_part.DATASET.SHARD_Y_IDX = j
            cfg_part.DATASET.TRAIN_FULL_SIZE = x_size
            cfg_part.DATASET.TEST_FULL_SIZE = y_size
            cfg_part.DATASET.TRAIN_SUBSET_START = start_x
            cfg_part.DATASET.TEST_SUBSET_START = start_y
            cfg_part.DATASET.TRAIN_SUBSET = min(x_size, min(x_size - start_x + x_shard_size, x_shard_size))
            cfg_part.DATASET.TEST_SUBSET = min(y_size, min(y_size - start_y + y_shard_size, y_shard_size))
            if cfg.KERNEL.COATESNG.ON:
                cn_cfg = cfg.KERNEL.COATESNG
                with open("/dev/shm/featurizer.pickle", "rb") as f:
                    net = pickle.load(f)
            else:
                net = None

            if use_aws_batch:
                data = {}
                data["net"] = net
                data["cfg"] = cfg_part
                data["dataset_A"] = dataset_a
                data["dataset_B"] = dataset_b
                data["net"] = net
                arg_uuid = m_repo.gen_short_uuid()
                data_bytes = pickle.dumps(data)
                client = boto3.client('s3')
                out_key = f"{PREFIX}/{arg_uuid}.pickle"
                out_bucket = BUCKET
                client.put_object(Bucket=out_bucket, Key=out_key, Body=data_bytes)
                job_name = f"{kernel_uuid}_{i}_{j}_{start_x}_{start_y}_symmetric_{symmetric}"
                job_cmd = f"python run_generate_kernel.py generate_kernel_remote {out_key} {out_bucket}"
                resp = submit_batch_job(job_name, job_cmd)
                job_id = resp['jobId']
                print(f"Batch Job {job_name}, Job ID: {job_id} submitted...")
                batch_job_ids[(i,j)] = resp['jobId']
            else:
                generate_kernel_only(cfg_part, dataset_a, dataset_b)
    if use_aws_batch:
        batch_client = boto3.client('batch', region_name="us-west-2")
        waiting, running, finished = batch_job_status(list(batch_job_ids.values()))
        #print("updating compute environment...")
        #batch_client.update_compute_environment(computeEnvironment="kernels3", computeResources={'desiredvCpus': len(waiting)*64})
        kernel_name = f"{dataset_a} x {dataset_b}"
        print(f"UPDATED!!! {kernel_name}")
        print("KERNEL NAME: ", kernel_name, "batch job ids", len(batch_job_ids))
        while len(finished) < len(batch_job_ids):
            waiting, running, finished = batch_job_status(list(batch_job_ids.values()))
            time.sleep(10)
            print(f"{kernel_name}: Waiting Jobs: {len(waiting)}, Running Jobs: {len(running)}, Finished Jobs: {len(finished)}")
    return kernel_uuid

def submit_batch_job(job_name, job_cmd):
    job_def_name = JOB_DEF_NAME 
    job_queue_name = JOB_QUEUE_NAME
    client = boto3.client('batch')
    container_override = {
            'command': job_cmd.split()
            }
    response = client.submit_job(
                jobDefinition=job_def_name,
                jobName=job_name,
                jobQueue=job_queue_name,
                containerOverrides=container_override)

    return response

def batch_job_status(job_ids):
    client = boto3.client('batch')
    waiting = {}
    running = {}
    finished = {}
    resp = client.describe_jobs(jobs=job_ids)
    for elem in resp['jobs']:
        if elem['status'] == 'SUBMITTED':
            waiting[elem['jobId']] = elem
        elif elem['status'] == 'PENDING':
            waiting[elem['jobId']] = elem
        elif elem['status'] == 'RUNNABLE':
            waiting[elem['jobId']] = elem
        elif elem['status'] == 'STARTING':
            waiting[elem['jobId']] = elem
        elif elem['status'] == 'RUNNING':
            running[elem['jobId']] = elem
        elif elem['status'] == 'FAILED':
            finished[elem['jobId']] = elem
        elif elem['status'] == 'SUCCEEDED':
            finished[elem['jobId']] = elem
    return waiting, running, finished

@cli.command('generate_kernel_remote')
@click.argument("cfg_key")
@click.argument("cfg_bucket")
def generate_kernel_remote(cfg_key, cfg_bucket):
    client = boto3.client('s3')
    data = pickle.loads(client.get_object(Bucket=cfg_bucket, Key=cfg_key)["Body"].read())
    cfg = data["cfg"]
    cfg.SYSTEM.USE_TQDM = False
    dataset_A = data["dataset_A"]
    dataset_B = data["dataset_B"]
    net = data["net"]
    if net is not None and cfg.KERNEL.COATESNG.ON:
        with open("/dev/shm/featurizer.pickle", "wb") as f:
            f.write(pickle.dumps(net))
    if cfg.SYSTEM.NUM_GPUS == 0:
        cfg.SYSTEM.NUM_GPUS = torch.cuda.device_count()
    return generate_kernel_only(cfg, dataset_A, dataset_B)


@cli.command('glue_kernels')
@click.argument("config-updates", nargs=-1)
@click.option("--config", default="configs/named_configs/myrtle11_cifar_1024_all_exp.yaml")
def glue_kernels(config_updates, config):
    ''' Given a kernel_uuid if kernel is fully materialized
        generate 1 kernel object that corresponds to entire kernel
    '''
    print("config updates", config_updates)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.merge_from_list(config_updates)
    return _glue_kernels(cfg)

def _glue_kernels(cfg):
    if cfg.SYSTEM.NUM_GPUS == 0:
        cfg.SYSTEM.NUM_GPUS = torch.cuda.device_count()

    m_repo = model_repository.ModelRepository()
    all_checkpoints = m_repo.get_checkpoints()
    valid_checkpoints = []
    for c in all_checkpoints:
        if c.extra_info is None: continue
        if c.extra_info.get("kernel_uuid") == cfg.DATASET.KERNEL_UUID:
            valid_checkpoints.append(c)

    shard_x_sizes = list(set([x.extra_info.get("shard_x_size") for x in valid_checkpoints]))
    shard_y_sizes = list(set([x.extra_info.get("shard_y_size") for x in valid_checkpoints]))

    x_datasets = list(set([x.extra_info.get("x_dataset") for x in valid_checkpoints]))
    y_datasets = list(set([x.extra_info.get("y_dataset") for x in valid_checkpoints]))

    x_sizes = list(set([x.extra_info.get("x_dataset_size") for x in valid_checkpoints]))
    y_sizes = list(set([x.extra_info.get("y_dataset_size") for x in valid_checkpoints]))

    print(shard_x_sizes)
    print(shard_y_sizes)
    assert len(shard_x_sizes) == 1
    assert len(shard_y_sizes) == 1
    assert len(x_sizes) == 1
    assert len(y_sizes) == 1

    assert len(x_sizes) == 1
    assert len(y_sizes) == 1

    dataset_A = x_datasets[0]
    dataset_B = y_datasets[0]

    shard_x_size = shard_x_sizes[0]
    shard_y_size = shard_y_sizes[0]

    x_size = x_sizes[0]
    y_size = y_sizes[0]
    print(x_size, y_size)

    print("dataset_A", dataset_A)
    print("dataset_B", dataset_B)
    dataset_A_obj = m_repo.get_dataset_by_name(dataset_A)
    dataset_B_obj = m_repo.get_dataset_by_name(dataset_B)

    print(dataset_A_obj, x_size)
    print(dataset_B_obj, y_size)

    shard_map = {}
    for cpt in valid_checkpoints:
        K_part = utils.bytes_to_numpy(m_repo.get_checkpoint_data(cpt.uuid, data_type="kernel"))
        shard_x_idx, shard_y_idx = (cpt.extra_info["shard_x_idx"], cpt.extra_info["shard_y_idx"])
        start_x, start_y = (cpt.extra_info["start_x"], cpt.extra_info["start_y"])
        end_x, end_y = (cpt.extra_info["end_x"], cpt.extra_info["end_y"])
        assert (shard_x_idx, shard_y_idx) not in shard_map
        shard_map[(shard_x_idx, shard_y_idx)] = (K_part, start_x, start_y, min(end_x, x_size - 1),  min(end_y, y_size - 1))
    print("Downloaded shards", sorted(list(shard_map.keys())))

    K = np.zeros((x_size, y_size))
    K.fill(float('-inf'))
    print("number of shard maps: ", len(shard_map.items()))
    for k, (K_part, start_x, start_y, end_x, end_y) in shard_map.items():
        print(start_x, start_y, end_x, end_y, K.shape, K_part.shape)
        # make sure parts are unique
        assert np.all(np.isinf(K[start_x:end_x+1, start_y:end_y+1]))
        print(end_x - start_x + 1, end_y - start_y + 1)
        K_part = K_part[:end_x - start_x + 1, :end_y - start_y + 1]
        K[start_x:min(end_x+1, K.shape[0]), start_y:min(end_y+1, K.shape[1])] = K_part
        if dataset_A == dataset_B :
            K[start_y:end_y+1, start_x:end_x+1] = K_part.T

    assert(np.all(np.isfinite(K)))
    uuid = m_repo.gen_short_uuid()
    print(f"storing kernel to mldb with kernel-uuid {uuid}...")
    model_obj = cpt.model

    extra_info = {}
    extra_info["kernel_only"] = True
    extra_info["kernel_name"] = cpt.extra_info["kernel_name"]
    extra_info["kernel_uuid"] = uuid
    extra_info["x_dataset"] = dataset_A_obj.name
    extra_info["y_dataset"] = dataset_B_obj.name
    extra_info["x_dataset_size"] = K.shape[0]
    extra_info["y_dataset_size"] = K.shape[1]
    extra_info["shard_x_idx"] = 0
    extra_info["shard_y_idx"] = 0
    extra_info["start_x"] = 0
    extra_info["end_x"] = int(K.shape[0]) - 1
    extra_info["start_y"] = 0
    extra_info["end_y"] = int(K.shape[1]) - 1
    extra_info["shard_x_size"] = K.shape[0]
    extra_info["shard_y_size"] = K.shape[1]
    cp_obj = m_repo.create_checkpoint(
        model_uuid=model_obj.uuid,
        kernel_data_bytes=utils.numpy_to_bytes(K),
        model_data_bytes=dill.dumps({}, protocol=4),
        extra_info=extra_info)
    m_repo.set_final_model_checkpoint(model_uuid=model_obj.uuid,
                                      checkpoint_uuid=cp_obj.uuid)
    print(f"Checkpoint UUID: {cp_obj.uuid}")
    return cp_obj.uuid









































def generate_kernel_only(cfg, dataset_A_name, dataset_B_name):
    m_repo = model_repository.ModelRepository()
    dataset_A = m_repo.get_dataset_by_name(dataset_A_name)
    dataset_B = m_repo.get_dataset_by_name(dataset_B_name)

    dataset_A_npz = np.load(
        io.BytesIO(m_repo.get_dataset_data(str(dataset_A.uuid))))
    dataset_B_npz = np.load(
        io.BytesIO(m_repo.get_dataset_data(str(dataset_B.uuid))))

    if "X_train" in set(dataset_A_npz.keys()):
        A = dataset_A_npz['X_train'].astype('float64')
    else:
        A = dataset_A_npz['X_test'].astype('float64')
    if "X_train" in set(dataset_B_npz.keys()):
        B = dataset_B_npz['X_train'].astype('float64')
    else:
        B = dataset_B_npz['X_test'].astype('float64')

    if cfg.DATASET.TRAIN_SUBSET is not None:
        A = A[cfg.DATASET.TRAIN_SUBSET_START:cfg.DATASET.TRAIN_SUBSET_START + cfg.DATASET.TRAIN_SUBSET]
        A_idxs = np.arange(cfg.DATASET.TRAIN_SUBSET_START,cfg.DATASET.TRAIN_SUBSET_START + cfg.DATASET.TRAIN_SUBSET)
    if cfg.DATASET.TEST_SUBSET is not None:
        B = B[cfg.DATASET.TEST_SUBSET_START:cfg.DATASET.TEST_SUBSET_START + cfg.DATASET.TEST_SUBSET]
        B_idxs = np.arange(cfg.DATASET.TEST_SUBSET_START,cfg.DATASET.TEST_SUBSET_START + cfg.DATASET.TEST_SUBSET)

    if cfg.SYSTEM.FLOAT_32:
        A = A.astype('float32')
        B = B.astype('float32')
    d_net = kernel_gen.GenericKernel(kernel_cfg=cfg.KERNEL,
                                     cache_path=cfg.SYSTEM.CACHE_PATH,
                                     float32=cfg.SYSTEM.FLOAT_32)
    arch_name = hash_arch(d_net.arch)
    arch_description = arch_short_name(d_net.arch)
    arch_obj = m_repo.get_architecture_by_name(arch_name, assert_exists=False)
    A_idxs_bio = io.BytesIO()
    np.save(A_idxs_bio, np.array(A_idxs))
    A_idxs_bytes = A_idxs_bio.getvalue()
    extra_info = {}
    extra_info["float32"] = cfg.SYSTEM.FLOAT_32
    if arch_obj is None:
        print(
            f"Arch id: {arch_name}, Arch shortname: {arch_description}  doesn't exist creating..."
        )
        arch_obj = m_repo.create_architecture(name=arch_name,
                                              description=arch_description,
                                              model_type="kernel",
                                              data_bytes=json.dumps(
                                                  cfg.KERNEL.ARCH).encode())
    else:
        print(
            f"Arch id: {arch_name}, Arch shortname: {arch_description} exists!"
        )

    ts = datetime.now().isoformat()
    model_name = f"{arch_name}_{cfg.DATASET.AUGMENTATIONS}_{cfg.DATASET.SHARD_X_IDX}_{cfg.DATASET.SHARD_Y_IDX}_{cfg.DATASET.KERNEL_UUID}_rand_seed_{cfg.DATASET.RAND_SEED}_{ts}"
    model_obj = m_repo.create_model(name=model_name,
                                    description=None,
                                    architecture_uuid=arch_obj.uuid,
                                    training_set_uuid=dataset_A.uuid,
                                    training_set_size=A.shape[0],
                                    training_set_indices_bytes=A_idxs_bytes)

    start = timer()
    kernel_name = f"{dataset_A.name}x{dataset_B.name}_{cfg.DATASET.AUGMENTATIONS}_{cfg.DATASET.SHARD_X_IDX}_rand_seed_{cfg.DATASET.RAND_SEED}_{cfg.DATASET.KERNEL_UUID}"
    if A.shape == B.shape:
        symmetric = np.allclose(A, B)
    else:
        symmetric = False
    
    K = kernel_gen.generate_kernel_parallel(
        cfg.KERNEL,
        A,
        B,
        num_gpus=cfg.SYSTEM.NUM_GPUS,
        symmetric=symmetric,
        batch_size=cfg.SYSTEM.BATCH_SIZE,
        model_uuid=model_obj.uuid,
        checkpoint_K=None,
        checkpoint_rows_done=None,
        cache_path=cfg.SYSTEM.CACHE_PATH,
        float32=cfg.SYSTEM.FLOAT_32,
        use_tqdm=cfg.SYSTEM.USE_TQDM,
        extra_info={"kernel_uuid": cfg.DATASET.KERNEL_UUID})
    end = timer()
    if cfg.DATASET.TRAIN_FULL_SIZE == 0:
        train_full_size = dataset_A.size
    else:
        train_full_size = cfg.DATASET.TRAIN_FULL_SIZE


    if cfg.DATASET.TEST_FULL_SIZE == 0:
        test_full_size = dataset_B.size
    else:
        test_full_size = cfg.DATASET.TEST_FULL_SIZE


    print(f"{A.shape[0]} x {B.shape[0]} {kernel_name} kernel took {end - start}")
    print("storing kernel to mldb...")
    extra_info["kernel_only"] = True
    extra_info["kernel_name"] = kernel_name
    extra_info["kernel_uuid"] = cfg.DATASET.KERNEL_UUID
    extra_info["x_dataset"] = dataset_A_name
    extra_info["y_dataset"] = dataset_B_name
    extra_info["x_dataset_size"] = train_full_size
    extra_info["y_dataset_size"] = test_full_size
    extra_info["shard_x_idx"] = cfg.DATASET.SHARD_X_IDX
    extra_info["shard_y_idx"] = cfg.DATASET.SHARD_Y_IDX
    extra_info["start_x"] = int(A_idxs[0])
    extra_info["end_x"] = int(A_idxs[-1])
    extra_info["start_y"] = int(B_idxs[0])
    extra_info["end_y"] = int(B_idxs[-1])
    extra_info["shard_x_size"] = int(cfg.DATASET.SHARD_X_SIZE)
    extra_info["shard_y_size"] = int(cfg.DATASET.SHARD_Y_SIZE)
    cp_obj = m_repo.create_checkpoint(
        model_uuid=model_obj.uuid,
        kernel_data_bytes=utils.numpy_to_bytes(K),
        model_data_bytes=dill.dumps({}, protocol=4),
        extra_info=extra_info)
    m_repo.set_final_model_checkpoint(model_uuid=model_obj.uuid,
                                      checkpoint_uuid=cp_obj.uuid)
    print(f"Checkpoint UUID: {cp_obj.uuid}")


if __name__ == "__main__":
    cli()
