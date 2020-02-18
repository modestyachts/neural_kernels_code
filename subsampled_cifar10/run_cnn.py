import numpy as np
import pickle
import gzip
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from enum import Enum
from core import *
from torch_backend import *
from argparse import ArgumentParser
import os

Loss = Enum('Loss', 'CROSSENTROPY MSE')
LR_Schedule = Enum('LR_Schedule', 'MYRTLE REDUCELRONPLATEAU')

os.system('sudo nvidia-persistenced')
os.system('sudo nvidia-smi -ac 877,1530')

def conv_bn(c_in, c_out, bn, bn_weight_init=1.0, affine=True, just_mean=False, instance_norm=False, num_groups=None, channels_per_group=None, **kw):
    if affine == False:
        bn_weight_init = None
    if bn:
        return {
            'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
            'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw), 
            'relu': nn.ReLU(True)
        }
    else:
        return {
            'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
            'relu': nn.ReLU(True)
        }

def basic_net(channels, weight, pool, bn, affine=True, just_mean=False, instance_norm=False, num_groups=None, channels_per_group=None, **kw):
    if max_pool:
        pooling = nn.MaxPool2d(4)
    else:
        pooling = nn.AvgPool2d(4)
    return {
        'prep': conv_bn(3, channels['prep'], bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw), pool=pool),
        'pool': pooling,
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], 10, bias=False),
        'classifier': Mul(weight),
    }

def parse_CP(description, d, weight, bn, affine=True, just_mean=False, instance_norm=False, num_groups=None, channels_per_group=None, **kw):
    if max_pool:
        pooling = nn.MaxPool2d(2)
    else:
        pooling = nn.AvgPool2d(2)
    n = {}
    layer = 0
    prev_channels = 3
    cur_channels = d
    total_pool = 1
    for char in description:
        layer = layer + 1
        if char == 'C':
            n[str(layer)] = conv_bn(prev_channels, cur_channels, bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw)
            prev_channels = cur_channels
        elif char == 'P':
            n[str(layer)] = pooling
            total_pool = 2 * total_pool
            cur_channels = 2 * cur_channels # Double the number of channels after each pooling layer. Note that this is not what Myrtle does, but it's similar
        else:
            print(f'unrecognized description character {char}')
    # At the end, pool down to 1
    remaining_pool = int(32 / total_pool)
    if max_pool:
        n[str(layer + 1)] = nn.MaxPool2d(remaining_pool)
    else:
        n[str(layer + 1)] = nn.AvgPool2d(remaining_pool)
    n['flatten'] = Flatten()
    n['linear'] = nn.Linear(prev_channels, 10, bias=False)
    n['classifier'] = Mul(weight)
    print(type(n))
    print(n)
    return n


def net(bn, description=None, channels=None, weight=0.125, pool=nn.AvgPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), affine=True, just_mean=False, instance_norm=False, num_groups=None, channels_per_group=None, **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    if description is not None:
        return parse_CP(description, d=channels['prep'], weight=weight, bn=bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw)
    residual = lambda c, **kw: {'in': Identity(), 'res1': conv_bn(c, c, bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw), 
                                'res2': conv_bn(c, c, bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw), 
                                'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')])}
    if max_pool:
        pool = nn.MaxPool2d(2)
    n = basic_net(channels, weight, pool, bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], bn, affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group, **kw)       
    print(type(n))
    print(n)
    return n

remove_identity_nodes = lambda net: remove_by_type(net, Identity)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def done(valid_accuracies, patience):
    epochs = len(valid_accuracies)
    if epochs < patience:
        return False
    if all(i <= valid_accuracies[-1*patience] for i in valid_accuracies[-1*patience:]):
        return True
    else:
        return False

def train(model, lr_schedule, max_lr, epochs, optimizer, train_set, test_set, batch_size, loss, logname, num_workers=0):
    if loss is Loss.MSE:
        train_batches = Batches(train_set, batch_size, shuffle=True, mse_loss=True, set_random_choices=True, num_workers=num_workers)
        test_batches = Batches(test_set, batch_size, shuffle=False, mse_loss=True, num_workers=num_workers)
    else:
        train_batches = Batches(train_set, batch_size, shuffle=True, set_random_choices=True, num_workers=num_workers)
        test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=num_workers)
    table, timer = TableLogger(), Timer()
    valid_accuracies = []
    if lr_schedule is LR_Schedule.MYRTLE:
        schedule = PiecewiseLinear([0, int(epochs/5), epochs], [0, max_lr, 0])
        optimizer.opt_params['lr'] = lambda step: schedule(step/len(train_batches))/batch_size
        for epoch in range(schedule.knots[-1]):
            epoch_stats = train_epoch(model, train_batches, test_batches, optimizer.step, None, timer, test_time_in_total=True) 
            valid_accuracies.append(epoch_stats['test acc'])
            lr = schedule(epoch+1)
            summary = union({'epoch': epoch+1, 'lr': lr}, epoch_stats)
            table.append(summary, logname=logname)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)
        epoch = 0
        while not done(valid_accuracies, patience=10):
            epoch_stats = train_epoch(model, train_batches, test_batches, optimizer.step, optimizer.zero_grad, timer, test_time_in_total=True) 
            valid_accuracies.append(epoch_stats['test acc'])
            scheduler.step(epoch_stats['test loss'])
            lr = get_lr(optimizer)
            summary = union({'epoch': epoch+1, 'lr': lr}, epoch_stats)
            table.append(summary, logname=logname)
            epoch = epoch + 1
    max_test_acc = max(valid_accuracies)
    print(f'max test accuracy is {max_test_acc}.')
    return max_test_acc



parser = ArgumentParser()
parser.add_argument('--num_trials', type=int, default=20, help='number of trials to loop over')
parser.add_argument('--d', type=int, default=64, help='dimension of the (first) hidden layer')
parser.add_argument('--no_bn', action='store_false', help='use this option to disable batch norm')
parser.add_argument('--bs', type=int, default=256, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='Nesterov momentum')
parser.add_argument('--loss', type=lambda loss: Loss[loss], default=Loss.MSE, help='CROSSENTROPY or MSE')
parser.add_argument('--lr_schedule', type=lambda lr_schedule: LR_Schedule[lr_schedule], default=LR_Schedule.MYRTLE, help='MYRTLE or REDUCELRONPLATEAU. The former runs for 60 epochs, and the latter runs until convergence.')
parser.add_argument('--epochs', type=int, default=400, help='number of training epochs, used only if lr_schedule is MYRTLE')
parser.add_argument('--max_lr', type=float, default=0.8, help='max learning rate')
parser.add_argument('--no_normalize_data', action='store_false', help='use this option to disable data normalization')
parser.add_argument('--no_zca', action='store_false', help='use this option to not preprocess data with ZCA whitening instead of just mean subtraction and stddev normalization')
parser.add_argument('--max_pool', action='store_true', help='use this option to replace avg pool with max pool')
parser.add_argument('--data_augmentation', action='store_true', help='use this option to enable data augmentation (flip, crop, and cutout)')
parser.add_argument('--training_size', type=int, default=10, help='how many examples to train with')
parser.add_argument('--testing_size', type=int, default=10000, help='how many examples to test with')
parser.add_argument('--allow_affine', action='store_true', help='use this option to include learnable parameters in batch norm')
parser.add_argument('--just_mean', action='store_true', help='use this option to undo the scaling portion of batch normalization')
parser.add_argument('--instance_norm', action='store_true', help='use this option to do instance norm instead of batch norm')
parser.add_argument('--num_groups', type=int, default=None, help='pass 0 to use instance norm, or an integer (factor of d) to use group norm with that many groups')
parser.add_argument('--channels_per_group', type=int, default=None, help='pass 1 to use instance norm, or an integer (factor of d) to use group norm with that many channels per group')
parser.add_argument('--description', type=str, default=None, help='pass a string of C and P where C is pool 3x3 + ReLU and P is pool 2x2')
args= parser.parse_args()

bs = args.bs
d = args.d
momentum = args.momentum
loss = args.loss
lr_schedule = args.lr_schedule
epochs = args.epochs
max_lr = args.max_lr
normalize_data = args.no_normalize_data
max_pool = args.max_pool
bn = args.no_bn
data_augmentation = args.data_augmentation
do_zca = args.no_zca
training_size = args.training_size
testing_size = args.testing_size
affine = args.allow_affine
just_mean = args.just_mean
instance_norm = args.instance_norm
num_groups = args.num_groups
channels_per_group = args.channels_per_group
description = args.description
patience = 3
num_trials = args.num_trials

params = f'{description}_d{d}_maxp{max_pool}_bn{bn}_inorm{instance_norm}_ngrps{num_groups}_cpergroup{channels_per_group}_bs{bs}_lr{lr_schedule.name}_max_lr{max_lr}_normal{normalize_data}_zca{do_zca}_epochs{epochs}_momentum{momentum}_loss{loss.name}_affinebn{affine}_jmean{just_mean}_maxpool{max_pool}_aug{data_augmentation}_trainsize{training_size}_testsize{testing_size}'
logname =f'logs/{params}.txt'
print(params)

losses = {
    'loss':  (nn.CrossEntropyLoss(reduce=False), [('classifier',), ('target',)]),
    'correct': (Correct(), [('classifier',), ('target',)]),
}
if loss is Loss.MSE:
    losses = {
        'loss':  (nn.MSELoss(reduce=False), [('classifier',), ('target',)]),
        'correct': (Correct_MSE(), [('classifier',), ('target',)]),
    }

DATA_DIR = './data'
dataset = cifar10(DATA_DIR)

if loss is Loss.MSE:
    y = torch.eye(10)
    dataset['train']['labels'] = y[dataset['train']['labels']]
    y = torch.eye(10)
    dataset['test']['labels'] = y[dataset['test']['labels']]


t = Timer()  
print('Preprocessing training data')
if data_augmentation: # also normalize if doing data augmentation
    train_set = list(zip(transpose(normalise(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
elif do_zca:
    train_data, test_data = zca(dataset['train']['data'], dataset['test']['data'])
    train_set = list(zip(transpose(train_data), dataset['train']['labels']))
    test_set = list(zip(transpose(test_data), dataset['test']['labels']))
elif normalize_data:
    train_set = list(zip(transpose(normalise(dataset['train']['data'])), dataset['train']['labels']))
else:
    train_set = list(zip(transpose(dataset['train']['data']), dataset['train']['labels']))
print(f'Finished in {t():.2} seconds')
print('Preprocessing test data')
if not do_zca:
    test_set = list(zip(transpose(normalise(dataset['test']['data'])), dataset['test']['labels']))
print(f'Finished in {t():.2} seconds')

num_classes = 10
train_set_0 = []
train_set_1 = []
train_set_2 = []
train_set_3 = []
train_set_4 = []
train_set_5 = []
train_set_6 = []
train_set_7 = []
train_set_8 = []
train_set_9 = []
for i in range(len(train_set)):
    current_item = train_set[i]
    current_label = current_item[1]
    if current_label[0] > 0:
        train_set_0.append(current_item)
    elif current_label[1] > 0:
        train_set_1.append(current_item)
    elif current_label[2] > 0:
        train_set_2.append(current_item)
    elif current_label[3] > 0:
        train_set_3.append(current_item)
    elif current_label[4] > 0:
        train_set_4.append(current_item)
    elif current_label[5] > 0:
        train_set_5.append(current_item)
    elif current_label[6] > 0:
        train_set_6.append(current_item)
    elif current_label[7] > 0:
        train_set_7.append(current_item)
    elif current_label[8] > 0:
        train_set_8.append(current_item)
    elif current_label[9] > 0:
        train_set_9.append(current_item)
test_set_0 = []
test_set_1 = []
test_set_2 = []
test_set_3 = []
test_set_4 = []
test_set_5 = []
test_set_6 = []
test_set_7 = []
test_set_8 = []
test_set_9 = []
for i in range(len(test_set)):
    current_item = test_set[i]
    current_label = current_item[1]
    if current_label[0] > 0:
        test_set_0.append(current_item)
    elif current_label[1] > 0:
        test_set_1.append(current_item)
    elif current_label[2] > 0:
        test_set_2.append(current_item)
    elif current_label[3] > 0:
        test_set_3.append(current_item)
    elif current_label[4] > 0:
        test_set_4.append(current_item)
    elif current_label[5] > 0:
        test_set_5.append(current_item)
    elif current_label[6] > 0:
        test_set_6.append(current_item)
    elif current_label[7] > 0:
        test_set_7.append(current_item)
    elif current_label[8] > 0:
        test_set_8.append(current_item)
    elif current_label[9] > 0:
        test_set_9.append(current_item)

max_test_acc = np.zeros(num_trials)
for i in range(num_trials):
    print(f'starting loop {i}')
    if training_size != len(train_set):
        train_subset_0_idx = np.random.choice(len(train_set_0), training_size // num_classes, replace=False)
        train_subset = list(train_set_0[i] for i in list(train_subset_0_idx))
        train_subset_1_idx = np.random.choice(len(train_set_1), training_size // num_classes, replace=False)
        train_subset.extend(list(train_set_1[i] for i in list(train_subset_1_idx)))
        train_subset_2_idx = np.random.choice(len(train_set_2), training_size // num_classes, replace=False)
        train_subset.extend(list(train_set_2[i] for i in list(train_subset_2_idx)))
        train_subset_3_idx = np.random.choice(len(train_set_3), training_size // num_classes, replace=False)
        train_subset.extend(list(train_set_3[i] for i in list(train_subset_3_idx)))
        train_subset_4_idx = np.random.choice(len(train_set_4), training_size // num_classes, replace=False)
        train_subset.extend(list(train_set_4[i] for i in list(train_subset_4_idx)))
        train_subset_5_idx = np.random.choice(len(train_set_5), training_size // num_classes, replace=False)
        train_subset.extend(list(train_set_5[i] for i in list(train_subset_5_idx)))
        train_subset_6_idx = np.random.choice(len(train_set_6), training_size // num_classes, replace=False)
        train_subset.extend(list(train_set_6[i] for i in list(train_subset_6_idx)))
        train_subset_7_idx = np.random.choice(len(train_set_7), training_size // num_classes, replace=False)
        train_subset.extend(list(train_set_7[i] for i in list(train_subset_7_idx)))
        train_subset_8_idx = np.random.choice(len(train_set_8), training_size // num_classes, replace=False)
        train_subset.extend(list(train_set_8[i] for i in list(train_subset_8_idx)))
        train_subset_9_idx = np.random.choice(len(train_set_9), training_size // num_classes, replace=False)
        train_subset.extend(list(train_set_9[i] for i in list(train_subset_9_idx)))
    else:
        train_subset = train_set
    if testing_size != len(test_set):
        test_subset_0_idx = np.random.choice(len(test_set_0), testing_size // num_classes, replace=False)
        test_subset = list(test_set_0[i] for i in list(test_subset_0_idx))
        test_subset_1_idx = np.random.choice(len(test_set_1), testing_size // num_classes, replace=False)
        test_subset.extend(list(test_set_1[i] for i in list(test_subset_1_idx)))
        test_subset_2_idx = np.random.choice(len(test_set_2), testing_size // num_classes, replace=False)
        test_subset.extend(list(test_set_2[i] for i in list(test_subset_2_idx)))
        test_subset_3_idx = np.random.choice(len(test_set_3), testing_size // num_classes, replace=False)
        test_subset.extend(list(test_set_3[i] for i in list(test_subset_3_idx)))
        test_subset_4_idx = np.random.choice(len(test_set_4), testing_size // num_classes, replace=False)
        test_subset.extend(list(test_set_4[i] for i in list(test_subset_4_idx)))
        test_subset_5_idx = np.random.choice(len(test_set_5), testing_size // num_classes, replace=False)
        test_subset.extend(list(test_set_5[i] for i in list(test_subset_5_idx)))
        test_subset_6_idx = np.random.choice(len(test_set_6), testing_size // num_classes, replace=False)
        test_subset.extend(list(test_set_6[i] for i in list(test_subset_6_idx)))
        test_subset_7_idx = np.random.choice(len(test_set_7), testing_size // num_classes, replace=False)
        test_subset.extend(list(test_set_7[i] for i in list(test_subset_7_idx)))
        test_subset_8_idx = np.random.choice(len(test_set_8), testing_size // num_classes, replace=False)
        test_subset.extend(list(test_set_8[i] for i in list(test_subset_8_idx)))
        test_subset_9_idx = np.random.choice(len(test_set_9), testing_size // num_classes, replace=False)
        test_subset.extend(list(test_set_9[i] for i in list(test_subset_9_idx)))
    else:
        test_subset = test_set

    n = net(bn=bn, description=description, channels={'prep': d, 'layer1': 2*d, 'layer2': 4*d, 'layer3': 8*d}, extra_layers=(), res_layers=(), affine=affine, just_mean=just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group)
    model = Network(union(n, losses)).to(device)
    opt = SGD(trainable_params(model), momentum=momentum, weight_decay=5e-4*bs, nesterov=True)
    if lr_schedule is LR_Schedule.REDUCELRONPLATEAU:
        opt = optim.SGD(model.parameters(), lr=max_lr, momentum=momentum)

    if data_augmentation:
        train_set_x = Transform(train_subset, [Crop(32, 32), FlipLR(), Cutout(8, 8)])
    else:
        train_set_x = Transform(train_subset, [])

    max_test_acc[i]=train(model, lr_schedule, max_lr, epochs, opt, train_set_x, test_subset, 
                batch_size=bs, loss=loss, logname=logname, num_workers=0)
print('list of max_test_acc')
print(max_test_acc)
print('mean max_test_acc')
print(np.mean(max_test_acc))
print('std max_test_acc')
print(np.std(max_test_acc))
print ('training size ', training_size)
print('done!')