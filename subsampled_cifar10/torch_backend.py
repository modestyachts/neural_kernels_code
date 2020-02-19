####################
# From https://github.com/davidcpage/cifar10-fast
####################

import numpy as np
import torch
from torch import nn
import torchvision
from core import build_graph, cat, to_numpy

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@cat.register(torch.Tensor)
def _(*xs):
    return torch.cat(xs)

@to_numpy.register(torch.Tensor)
def _(x):
    return x.detach().cpu().numpy()  

def warmup_cudnn(model, batch_size):
    #run forward and backward pass of the model on a batch of random inputs
    #to allow benchmarking of cudnn kernels 
    batch = {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).cuda().float(), 
        'target': torch.LongTensor(np.random.randint(0,10,batch_size)).cuda()
    }
    model.train(True)
    o = model(batch)
    o['loss'].sum().backward()
    model.zero_grad()
    torch.cuda.synchronize()


#####################
## dataset
#####################

# Python 3.6 (AWS)
# def cifar10(root):
#     train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
#     test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
#     return {
#         'train': {'data': train_set.train_data, 'labels': train_set.train_labels},
#         'test': {'data': test_set.test_data, 'labels': test_set.test_labels}
#     }

# Python 3.7
def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, mse_loss=False, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.mse_loss = mse_loss
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        if self.mse_loss:
            return ({'input': x.to(device).float(), 'target': y.to(device).float()} for (x,y) in self.dataloader)
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)

#####################
## torch stuff
#####################

class Identity(nn.Module):
    def forward(self, x): return x
    
class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight
    
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Add(nn.Module):
    def forward(self, x, y): return x + y 
    
class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)
    
class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

class Correct_MSE(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim=1)[1] == target.max(dim=1)[1]

class NormLayer(nn.Module):
    # num_groups and channels_per_group both control group norm. Only one of these should not be None. 
    # num_groups=0 and channels_per_group=1 are aliases for instance_norm=True. 
    def __init__(self, num_features, eps=1e-5, momentum=0.1, mean=True, norm=True, instance_norm=False, num_groups=None, channels_per_group=None):
        super(NormLayer, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.mean = mean
        self.norm = norm
        self.eps = eps
        self.instance_norm = instance_norm
        self.num_groups = num_groups
        self.channels_per_group = channels_per_group
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def forward(self, input_):
        batchsize, channels, height, width = input_.size()
        if self.instance_norm or self.num_groups == 0 or self.channels_per_group == 1:
            numel = height * width
            input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, batchsize, numel)
            sum_ = input_.sum(2)
            sum_of_square = input_.pow(2).sum(2)
        elif self.num_groups is None and self.channels_per_group is None:
            numel = batchsize * height * width
            input_ = input_.permute(1, 0, 2, 3).contiguous().view(channels, numel)
            sum_ = input_.sum(1)
            sum_of_square = input_.pow(2).sum(1)
        else: # Do group norm
            if self.num_groups is not None:
                group_size = channels // self.num_groups
                if group_size != channels / self.num_groups:
                    print(f'The number of channels {channels} is not a multiple of the number of groups {self.num_groups}.')
            elif self.channels_per_group is not None:
                group_size = self.channels_per_group
                if channels // group_size != channels / group_size:
                    print(f'The number of channels {channels} is not a multiple of the number of channels per group {group_size}')
            numel = height * width * group_size
            input_ = input_.contiguous().view(batchsize, channels // group_size, numel)
            sum_ = input_.sum(2)
            sum_of_square = input_.pow(2).sum(2)
        mean = sum_ / numel
        sumvar = sum_of_square - sum_ * mean

        unbias_var = sumvar / (numel - 1)
        bias_var = sumvar / numel
        inv_std = 1 / (bias_var + self.eps).pow(0.5)
        output = input_
        if self.mean:
            if self.instance_norm or self.num_groups is not None or self.channels_per_group is not None:
                output = output - mean.unsqueeze(2)
            else:
                output = output - mean.unsqueeze(1)
        if self.norm:
            if self.instance_norm or self.num_groups is not None or self.channels_per_group is not None:
                output = output * inv_std.unsqueeze(2)
            else:
                output = output * inv_std.unsqueeze(1)
        if (self.num_groups is not None and self.num_groups > 0) or (self.channels_per_group is not None and self.channels_per_group > 1): # Group norm
            return output.view(batchsize, channels, height, width).contiguous()
        return output.view(channels, batchsize, height, width).permute(1, 0, 2, 3).contiguous() # Instance norm or batch norm

def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False, affine=True, just_mean=False, instance_norm=False, num_groups=None, channels_per_group=None):
    m = nn.BatchNorm2d(num_channels, affine=affine)
    if instance_norm or num_groups is not None or channels_per_group is not None: # Group/Instance norm acts the same during training and testing
        m = NormLayer(num_channels, norm=not just_mean, instance_norm=instance_norm, num_groups=num_groups, channels_per_group=channels_per_group)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False
    return m

class Network(nn.Module):
    def __init__(self, net):
        self.graph = build_graph(net)
        super().__init__()
        for n, (v, _) in self.graph.items(): 
            setattr(self, n, v)

    def forward(self, inputs):
        self.cache = dict(inputs)
        for n, (_, i) in self.graph.items():
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache
    
    def half(self):
        for module in self.children():
            if not isinstance(module, nn.BatchNorm2d):
                module.float()    
        return self

trainable_params = lambda model:filter(lambda p: p.requires_grad, model.parameters())

class TorchOptimiser():
    def __init__(self, weights, optimizer, step_number=0, **opt_params):
        self.weights = weights
        self.step_number = step_number
        self.opt_params = opt_params
        self._opt = optimizer(weights, **self.param_values())
    
    def param_values(self):
        return {k: v(self.step_number) if callable(v) else v for k,v in self.opt_params.items()}
    
    def step(self):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._opt.step()

    def zero_grad(self):
        self._opt.zero_grad()

    def __repr__(self):
        return repr(self._opt)
        
def SGD(weights, lr=0, momentum=0, weight_decay=0, dampening=0, nesterov=False):
    return TorchOptimiser(weights, torch.optim.SGD, lr=lr, momentum=momentum, 
                          weight_decay=weight_decay, dampening=dampening, 
                          nesterov=nesterov)
