import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import time
import gc
import argparse
import torch
import os
import sys
import signal

# I hate multiprocessing
from torch.multiprocessing import Queue, Process, Value
from queue import Empty

import tc_kernels as tck
import utils
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from timeit import default_timer as timer
from config import get_cfg_defaults
import multiprocessing as mp
import pickle

INTERNAL_CHUNK_SIZE = 8

PROCESS_MAP = {}
class GenericKernel(object):
    def __init__(self, *, kernel_cfg, cache_path="tc_cache", float32=False, hps={}):
        self.float32 = float32
        self.tcw = tck.TCWrapper(cache_path, float32=float32)
        self.hps = hps
        self.kernel_cfg = kernel_cfg
        self.arch = kernel_cfg.ARCH
        self.defaults = dict(kernel_cfg.ARCH_DEFAULTS)
        for layer in self.arch:
            lyr_name = layer["layer"]
            if lyr_name in self.defaults:
                for key, val in self.defaults[lyr_name].items():
                    if key not in layer:
                        layer[key] = val
        self.layer_map = {"pool8": self.tcw.pool8, "pool7": self.tcw.pool7, "pool4": self.tcw.pool4,  "pool2": self.tcw.pool2, "pool4": self.tcw.pool4,
                "conv3": self.tcw.conv3, "conv3zp": self.tcw.conv3zp, "conv3zpinorm": self.tcw.conv3zp_inorm, "pool32": self.tcw.pool32, "pool30": self.tcw.pool30,
                "relu": self.tcw.relu, "relu_project": self.tcw.relu_project, "quartic": self.tcw.quartic, "quartic_project": self.tcw.quartic_project,
                "quadratic": self.tcw.quadratic, "quadratic_project": self.tcw.quadratic_project, "exponential": self.tcw.exponential, "exponential_shifted": self.tcw.exponential_shifted,
                "group_norm_32": self.tcw.group_norm_32, "group_norm_16": self.tcw.group_norm_16,
                "group_norm_8": self.tcw.group_norm_8, "group_norm_4": self.tcw.group_norm_4,  "exponential_shifted_project": self.tcw.exponential_shifted_project}
        self.input_layer_map = {"conv3": self.tcw.conv3_input, "conv3zp": self.tcw.conv3zp_input, "conv3zpinorm": self.tcw.conv3zpinorm_input, "input": self.tcw.input, "conv3zp_all_chan": self.tcw.conv3zp_all_chan_input}
        self.layers, self.connections, self.kwargs_list = self.build(self.arch)
        self.residual_memory = {}
        for k,v in self.connections.items():
            self.residual_memory[v] = None
        print("Layer KWARGS:", self.kwargs_list)
    def build(self, arch):
        network = []
        kwargs = []
        connections = {}
        l = arch[0]
        if l['layer'] not in self.input_layer_map:
            raise Exception("Unknown Input layer")
        else:
            network.append(self.input_layer_map[l['layer']])
            lc = l.copy()
            lc.pop("layer")
            kwargs.append(lc)
        for i, l in enumerate(arch[1:]):
            if l['layer'] not in self.layer_map:
                raise Exception(f"Unknown layer: {l['layer']}")
            else:
                network.append(self.layer_map[l['layer']])
                lc = l.copy()
                lc.pop("layer")
                if "residual" in lc:
                    connections[i+1] = lc.pop("residual")
                kwargs.append(lc)
        return network, connections, kwargs

    def forward(self, X_batch, Y_batch, gpu=0, pp_net=None):
        start = time.time()
        K = np.zeros((X_batch.shape[0], Y_batch.shape[0]))
        num_x = X_batch.shape[0]
        num_y = Y_batch.shape[0]

        if num_x < num_y:
            X_batch_new = torch.zeros(*Y_batch.shape)
            X_batch_new[:num_x] = X_batch
            X_batch = X_batch_new
        elif num_y < num_x:
            Y_batch_new = torch.zeros(*X_batch.shape)
            Y_batch_new[:num_y] = Y_batch
            Y_batch = Y_batch_new

        assert X_batch.shape == Y_batch.shape
        all_norms_x = []
        all_norms_y = []
        x_bs = []
        y_bs = []
        N = X_batch.shape[0]
        M = Y_batch.shape[0]
        if self.float32:
            default_precision = "float32"
        else:
            default_precision = "float64"
        with torch.cuda.device(gpu):
            K = torch.zeros((N, M))
            if default_precision == "float64":
                K = K.double()
            assert len(X_batch.shape) == 4
            assert len(Y_batch.shape) == 4
            all_norms = [all_norms_x, all_norms_y]
            if pp_net is not None:
                x_cn = X_batch.permute(0,3,1,2).contiguous().cuda()
                x_lift = pp_net._forward(x_cn).permute(0,2,3,1).contiguous()
                y_cn = Y_batch.permute(0,3,1,2).contiguous().cuda()
                y_lift = pp_net._forward(y_cn).permute(0,2,3,1).contiguous()
                X_batch = x_lift
                Y_batch = y_lift
            else:
                X_batch = X_batch.cuda()
                Y_batch = Y_batch.cuda()

            for i_chunk in utils.chunks(range(X_batch.shape[0]), INTERNAL_CHUNK_SIZE):
                x_b = X_batch[i_chunk, :]
                x_bs.append(x_b)
                precision = self.kwargs_list[0].get("precision", default_precision)
                with self.tcw.precision(precision):
                    prev_norm = self.layers[0](x_b, x_b, **self.kwargs_list[0])
                all_norms_x.append([])
                all_norms_x[-1].append(prev_norm)
                for i,layer in enumerate(self.layers[1:]):
                    if i+1 in self.connections:
                        prev_norm = prev_norm + all_norms_x[-1][self.connections[i+1]]
                    precision = self.kwargs_list[i+1].get("precision", default_precision)
                    with self.tcw.precision(precision):
                        prev_norm = layer(prev_norm, prev_norm, prev_norm, **self.kwargs_list[i+1])
                    if self.kwargs_list[i+1].get("store_norm", True):
                        all_norms_x[-1].append(prev_norm)
                    else:
                        all_norms_x[-1].append(None)
            for j_chunk in utils.chunks(range(Y_batch.shape[0]), INTERNAL_CHUNK_SIZE):
                y_b = Y_batch[j_chunk, :]
                y_bs.append(y_b)
                precision = self.kwargs_list[0].get("precision", default_precision)
                with self.tcw.precision(precision):
                    prev_norm = self.layers[0](y_b, y_b,  **self.kwargs_list[0])
                all_norms_y.append([])
                all_norms_y[-1].append(prev_norm)
                for i,layer in enumerate(self.layers[1:]):
                    if i+1 in self.connections:
                        prev_norm = prev_norm + all_norms_y[-1][self.connections[i+1]]
                    precision = self.kwargs_list[i+1].get("precision", default_precision)
                    with self.tcw.precision(precision):
                        prev_norm = layer(prev_norm, prev_norm, prev_norm, **self.kwargs_list[i+1])
                    if self.kwargs_list[i+1].get("store_norm", True):
                        all_norms_y[-1].append(prev_norm)
                    else:
                        all_norms_y[-1].append(None)

            x_bytes = sum([sum([np.product(x.shape) for x in y if x is not None])*8 for y in all_norms_x])
            y_bytes = sum([sum([np.product(x.shape) for x in y if x is not None])*8 for y in all_norms_y])
            for i_idx, i_chunk in enumerate(utils.chunks(range(X_batch.shape[0]), 8)):
                x_b = x_bs[i_idx]
                for j_idx, j_chunk in enumerate(utils.chunks(range(Y_batch.shape[0]), 8)):
                    x_norms = all_norms_x[i_idx]
                    y_norms = all_norms_y[j_idx]
                    y_b = y_bs[j_idx]
                    precision = self.kwargs_list[0].get("precision", default_precision)
                    with self.tcw.precision(precision):
                        Kxy_0 = self.layers[0](x_b, y_b, **self.kwargs_list[0])
                    prev_K = Kxy_0
                    if 0 in self.residual_memory:
                        self.residual_memory[0] = prev_K
                    for i, layer in enumerate(self.layers[1:]):
                        x_norm = x_norms[i]
                        y_norm = y_norms[i]
                        precision = self.kwargs_list[i+1].get("precision", default_precision)

                        if i+1 in self.connections:
                            prev_K = prev_K + self.residual_memory[self.connections[i+1]]
                            if x_norm is not None:
                                x_norm = x_norm + x_norms[self.connections[i+1]]
                            if y_norm is not None:
                                y_norm = y_norm + y_norms[self.connections[i+1]]

                        # if x_norm is None this means the next layer does not need it
                        # so we can pass in a dummy value
                        if x_norm is None:
                            x_norm = prev_K
                        if y_norm is None:
                            y_norm = prev_K

                        with self.tcw.precision(precision):
                            prev_K = layer(prev_K, x_norm, y_norm, **self.kwargs_list[i+1])
                            #print(layer, precision, prev_K.dtype)

                        if i+1 in self.residual_memory:
                            self.residual_memory[i+1] = prev_K

                    start_i = min(i_chunk)
                    end_i = max(i_chunk)

                    start_j = min(j_chunk)
                    end_j = max(j_chunk)

                    K[start_i:end_i+1, start_j:end_j+1] = prev_K.squeeze().cpu()

        # unpad
        if num_x < num_y:
            return K[:num_x, :]
        elif num_y < num_x:
            return K[:, :num_y]
        else:
            return K

def _symmetric_fill(K, x, y, batch_size):
    x_idxs = torch.arange(x.shape[0])
    y_idxs = torch.arange(y.shape[0])

    x_data = TensorDataset(x_idxs, torch.from_numpy(x))
    x_loader = DataLoader(x_data, batch_size=batch_size)

    y_data = TensorDataset(y_idxs, torch.from_numpy(y))
    y_loader = DataLoader(y_data, batch_size=batch_size)
    for batch_ndx, (x_idxs, x_b) in enumerate(x_loader):
        x_idxs = x_idxs.numpy().astype('int')
        for batch_ndx, (y_idxs, y_b) in enumerate(y_loader):
            y_idxs = y_idxs.numpy().astype('int')
            start_x = min(x_idxs)
            end_x = max(x_idxs) + 1
            start_y = min(y_idxs)
            end_y = max(y_idxs) + 1
            if start_y > start_x:
                # only calculate lower triangle
                continue
            K[start_x:end_x, start_y:end_y] = K[start_y:end_y, start_x:end_x].T
    return K


def generate_kernel(dnet, x, y, batch_size=16, symmetric=False, cache_path="tc_cache", float32=False, extra_info={}):
    ''' Takes in two numpy arrays x and y that are N x H x W x C and M x H x W x C
        and spits out a kernel matrix K that is N x M
    '''

    assert dnet.float32 == float32
    #TODO fixme
    N = x.shape[0]
    M = y.shape[0]
    x_idxs = torch.arange(x.shape[0])
    y_idxs = torch.arange(y.shape[0])

    x_data = TensorDataset(x_idxs, torch.from_numpy(x))
    x_loader = DataLoader(x_data, batch_size=batch_size)

    y_data = TensorDataset(y_idxs, torch.from_numpy(y))
    y_loader = DataLoader(y_data, batch_size=batch_size)

    if symmetric:
        assert np.all(x == y)

    K = np.memmap("/dev/shm/kernel", mode="w+", dtype="float64", shape=(N, M))
    K.fill(np.inf)
    rows_done = np.memmap("/dev/shm/rowsdone", mode="w+", dtype="uint16", shape=(1,))
    if checkpoint_rows_done is not None:
        rows_done[:] = np.copy(utils.bytes_to_numpy(checkpoint_rows_done))
        K[:rows_done[0],:] = np.copy(utils.bytes_to_numpy(checkpoint_K))

    processes = []
    last_checkpoint = N*M
    work_left = N*M
    n = 0
    num_column_blocks = int(N/batch_size)

    with tqdm(total=N*M) as pbar:
        for batch_ndx, (x_idxs, x_b) in enumerate(x_loader):
            x_idxs = x_idxs.numpy().astype('int')
            for batch_ndx, (y_idxs, y_b) in enumerate(y_loader):
                y_idxs = y_idxs.numpy().astype('int')
                start_x = min(x_idxs)
                end_x = max(x_idxs) + 1

                start_y = min(y_idxs)
                end_y = max(y_idxs) + 1

                if start_x > start_y and symmetric:
                    # only calculate upper triangle
                    continue
                if end_x <= rows_done[0]:
                    # already calculated, move to next
                    pbar.update(batch_size*batch_size)
                    work_left -= batch_size*batch_size
                    if symmetric and start_x != end_x:
                        pbar.update(batch_size*batch_size)
                        work_left -= batch_size*batch_size
                    continue

                kx = dnet.forward(x_b, y_b).cpu().numpy().squeeze()
                K[start_x:end_x, start_y:end_y] = kx
                pbar.update(batch_size*batch_size)
                work_left -= batch_size*batch_size
                if symmetric and start_x != start_y:
                    pbar.update(batch_size*batch_size)
                    work_left -= batch_size*batch_size
                n += 1
    for p in processes:
        p.join()

    if symmetric:
        _symmetric_fill(K, x, y, batch_size)

    print(f"Total .forward() calls: {n}")
    K_copy = np.zeros(K.shape)
    np.copyto(K_copy, K)
    assert np.all(np.isfinite(K_copy))
    return K_copy

def _kernel_gen_help(done_q, data_q, kernel_cfg, batch_size, symmetric, gpu_idx, shape_K, cache_path, float32, done, verbose):
    print("STARTING KERNEL GEN HELP")
    if not verbose:
        pass
        #sys.stdout = open(os.devnull, 'w')
    if float32:
        K = np.memmap("/dev/shm/kernel", mode="r+", dtype="float32", shape=shape_K)
    else:
        K = np.memmap("/dev/shm/kernel", mode="r+", dtype="float64", shape=shape_K)
    dnet = GenericKernel(kernel_cfg=kernel_cfg, cache_path=cache_path, float32=float32)
    n = 0
    if kernel_cfg.COATESNG.ON:
        with open("/dev/shm/featurizer.pickle", "rb") as f:
            with torch.cuda.device(gpu_idx):
                net = pickle.load(f)
                net.activate(0, kernel_cfg.COATESNG.NUM_FILTERS)
                net = net.cuda()
    else:
        net = None

    while True:
        if done.value > 0:
            print("DONE!")
            break
        try:
            (x_idxs, x_b), (y_idxs, y_b) = data_q.get(timeout=10)
            x_idxs = x_idxs.numpy().astype('int')
            y_idxs = y_idxs.numpy().astype('int')
            start_x = min(x_idxs)
            end_x = max(x_idxs) + 1
            start_y = min(y_idxs)
            end_y = max(y_idxs) + 1
            if start_x > start_y and symmetric:
                # only calculate upper triangle
                continue
            kx = dnet.forward(x_b, y_b, gpu=gpu_idx, pp_net=net).cpu().numpy().squeeze()
            K[start_x:end_x, start_y:end_y] = kx
            done_q.put(len(x_idxs)*len(y_idxs))
            if symmetric and start_x != start_y:
                done_q.put(len(x_idxs)*len(y_idxs))
            n += 1
        except Empty:
            break
    print(f"gpu {gpu_idx} called forward {n} times")



def generate_kernel_parallel(kernel_cfg, x, y, batch_size=32, num_gpus=4, symmetric=False, model_uuid=None, checkpoint_K=None, checkpoint_rows_done=None, cache_path="tc_cache", float32=False, extra_info={}, verbose=False, use_tqdm=True):
    ''' Takes in two numpy arrays x and y that are N x H x W x C and M x H x W x C
        and spits out a kernel matrix K that is N x M
    '''

    #TODO fixme
    print("Batch Size ", batch_size)
    assert num_gpus <= torch.cuda.device_count()
    N = x.shape[0]
    M = y.shape[0]
    if float32:
        K = np.memmap("/dev/shm/kernel", mode="w+", dtype="float32", shape=(N, M))
    else:
        K = np.memmap("/dev/shm/kernel", mode="w+", dtype="float64", shape=(N, M))

    K.fill(np.inf)
    rows_done = np.memmap("/dev/shm/rowsdone", mode="w+", dtype="uint16", shape=(1,))
    if checkpoint_rows_done is not None:
        rows_done[:] = np.copy(utils.bytes_to_numpy(checkpoint_rows_done))
        K[:rows_done[0],:] = np.copy(utils.bytes_to_numpy(checkpoint_K))

    n = 0
    done_q = Queue()
    data_q = Queue()
    done = Value('i', 0)
    num_column_blocks = int(N/batch_size)

    x_idxs = torch.arange(x.shape[0])
    y_idxs = torch.arange(y.shape[0])

    x_data = TensorDataset(x_idxs, torch.from_numpy(x))
    x_loader = DataLoader(x_data, batch_size=batch_size)

    y_data = TensorDataset(y_idxs, torch.from_numpy(y))
    y_loader = DataLoader(y_data, batch_size=batch_size)

    processes = []

    x_data = [x for x in x_loader]
    y_data = [y for y in y_loader]
    count = 0
    start_time = time.time()
    for x_idxs, x_b in x_data:
        for y_idxs, y_b in y_data:
            count += 1
            start_x = int(min(x_idxs))
            end_x = int(max(x_idxs) + 1)
            start_y = int(min(y_idxs))
            end_y = int(max(y_idxs) + 1)
            if end_x > rows_done[0]:
                data_q.put(((x_idxs, x_b), (y_idxs, y_b)))
            #print(start_x, start_y)
            if count % 1000 == 0:
                print("Current Count Is: ", count)
    os.environ["OMP_NUM_THREADS"] = str(1)

    for gpu_idx in range(num_gpus):
        p = Process(target=_kernel_gen_help, args=(done_q, data_q, kernel_cfg,  batch_size, symmetric, gpu_idx, K.shape, cache_path, float32, done, verbose))
        processes.append(p)

    for i,p in enumerate(processes):
        p.start()
    if symmetric:
        done_work = rows_done[0]*M + (N-rows_done[0])*(rows_done[0])
    else:
        done_work = rows_done[0]*M
    work_left = N*M - done_work
    last_checkpoint = work_left
    print("Data_q size start", data_q.qsize())
    if use_tqdm:
        pbar = tqdm(total=N*M)
    else:
        pbar = None
    total_progress = 0
    while work_left > 0:
        progress = done_q.get()
        total_progress += progress
        work_left -= progress
        elapsed = time.time() - start_time
        avg_speed = total_progress/elapsed
        time_left = utils.pretty_time_delta(work_left/avg_speed)
        if pbar is not None:
            pbar.update(progress)
        else:
            print(f"Work Left: {work_left}, Work done so far: {total_progress}, Time Left: {time_left}")
    if pbar is not None:
        pbar.close()
    print("Data_q size end", data_q.qsize())
    done.value = 1
    for i,p in enumerate(processes):
        p.join()
    np.save("/tmp/K_train_full.npy", K)
    if symmetric:
        _symmetric_fill(K, x, y, batch_size)
    K_copy = np.zeros(K.shape)
    np.copyto(K_copy, K)
    assert np.all(np.isfinite(K_copy))
    return K_copy



def main():
    N = 256
    H = 32
    W = 32
    x = np.random.randn(N, H, W, 3)
    y = np.random.randn(N, H, W, 3)
    cfg = get_cfg_defaults()
    dnet = GenericKernel(kernel_cfg=cfg.KERNEL)
    start = timer()
    K = generate_kernel(dnet, x, y)
    end = timer()
    print(f"{N} x {N} kernel took {end - start} seconds")


if __name__ == "__main__":
    main()
