try:
    import tensor_comprehensions as tc
except:
    print("Warning TensorComprehensions not found...")
import contextlib
import torch
import numpy as np
from timeit import default_timer as timer
import torch.nn.functional as F
import re
import math
import argparse
import re
count = 0
'''
================== DOUBLE IMPLEMENTATIONS ========================
'''

template_kernel_conv =\
'''
def conv<conv_size>_input(double(B,H,W,P) X, double(B,H,W,P) Y) -> (conv_output) {
    conv_output(n,m,i,j,k,l) +=! (X(n,i + wx, j + wy, c) * Y(m,k + wx, l + wy, c))  where wx in 0:<conv_size>, wy in 0:<conv_size>
    }
'''

template_kernel_conv_all_chan  =\
'''
def conv<conv_size>_all_chan_input(double(B,H,W,P) X, double(B,H,W,P) Y) -> (conv_output) {
    conv_output(n,m,i,j,k,l) +=! (X(n,i + wx, j + wy, c0) * Y(m,k + wx, l + wy, c1))  where wx in 0:<conv_size>, wy in 0:<conv_size>
    }
'''


kernel_input =\
'''
def input(double(B,H,W,P) X, double(B,H,W,P) Y) -> (conv_output) {
    conv_output(n,m,i,j,k,l) +=! X(n,i,j,c) * Y(m,k,l,c)
    }
'''

kernel_float_input =\
'''
def input(float(B,H,W,P) X, float(B,H,W,P) Y) -> (conv_output) {
    conv_output(n,m,i,j,k,l) +=! X(n,i,j,c) * Y(m,k,l,c)
    }
'''

kernel_conv3_input = template_kernel_conv.replace("<conv_size>", "3")
kernel_conv3_all_chan_input = template_kernel_conv_all_chan.replace("<conv_size>", "3")
kernel_conv6_input = template_kernel_conv.replace("<conv_size>", "6")

template_kernel_conv_layer =\
'''
def conv<conv_size>_layer(double(BX,BY,H,W,H,W) K, double(BX, BY,H,W,H,W) Kxx, double(BX,BY,H,W,H,W) Kyy) -> (conv_output) {
    conv_output(n,m,i,j,k,l) +=! K(n,m, i + wx, j + wy, k + wx, l + wy)  where wx in 0:<conv_size>, wy in 0:<conv_size>
}
'''

kernel_conv3_layer = template_kernel_conv_layer.replace("<conv_size>", "3")
kernel_conv5_layer = template_kernel_conv_layer.replace("<conv_size>", "5")
kernel_conv6_layer = template_kernel_conv_layer.replace("<conv_size>", "6")

template_kernel_pool_layer =\
'''
def pool<pool_size>(double(BX,BY,H,W,H,W) K, double(BX, BY,H,W,H,W) Kxx, double(BX,BY,H,W,H,W) Kyy) -> (pool_output) {
    pool_output(n,m,i,j,k,l) +=!  K(n,m, <pool_stride>*i + wx, <pool_stride>*j + wy, <pool_stride>*k + wz, <pool_stride>*l + wl)  where wx in 0:<pool_size>, wy in 0:<pool_size>, wz in 0:<pool_size>, wl in 0:<pool_size>
    pool_output(n,m,i,j,k,l)  = pool_output(n,m,i,j,k,l)
}
'''

kernel_pool2  = template_kernel_pool_layer.replace("<pool_size>", "2").replace("<pool_stride>", "2")
kernel_pool3  = template_kernel_pool_layer.replace("<pool_size>", "3").replace("<pool_stride>", "3")
kernel_pool4  = template_kernel_pool_layer.replace("<pool_size>", "4").replace("<pool_stride>", "4")
kernel_pool7 = template_kernel_pool_layer.replace("<pool_size>", "7").replace("<pool_stride>", "7")
kernel_pool8 = template_kernel_pool_layer.replace("<pool_size>", "8").replace("<pool_stride>", "8")
kernel_pool32  = template_kernel_pool_layer.replace("<pool_size>", "32").replace("<pool_stride>", "32")
kernel_pool30 = template_kernel_pool_layer.replace("<pool_size>", "30").replace("<pool_stride>", "30")
kernel_pool32 = template_kernel_pool_layer.replace("<pool_size>", "32").replace("<pool_stride>", "32")


kernel_relu =\
'''
def relu(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy) -> (conv_output_scaled, conv_output_scaled_sq, conv_relu_output) {
    conv_output_scaled(n,m,i,j,k,l)  = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    conv_output_scaled_sq(n,m,i,j,k,l) = conv_output_scaled(n,m,i,j,k,l) * conv_output_scaled(n,m,i,j,k,l)
    conv_relu_output(n,m,i,j,k,l) = sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)) * (1/<pi> * sqrt(1 - conv_output_scaled_sq(n,m,i,j,k,l)) + conv_output_scaled(n,m,i,j,k,l)* (1 - (1/<pi>) * acos(conv_output_scaled(n,m,i,j,k,l))))
}
'''

kernel_relu_project =\
'''
def relu_project(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy) -> (conv_output_scaled, conv_output_scaled_sq, conv_relu_output) {
    conv_output_scaled(n,m,i,j,k,l)  = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    conv_output_scaled_sq(n,m,i,j,k,l) = conv_output_scaled(n,m,i,j,k,l) * conv_output_scaled(n,m,i,j,k,l)
    conv_relu_output(n,m,i,j,k,l) = (1/<pi> * sqrt(1 - conv_output_scaled_sq(n,m,i,j,k,l)) + conv_output_scaled(n,m,i,j,k,l)* (1 - (1/<pi>) * acos(conv_output_scaled(n,m,i,j,k,l))))
}
'''

kernel_quartic =\
'''
def quartic(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy) -> (K2, K3, K4) {
    K2(n,m,i,j,k,l)  = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l)
    K4(n,m,i,j,k,l) = sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)) *  K3(n,m,i,j,k,l)
}
'''

kernel_quartic_project =\
'''
def quartic_project(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy) -> (K2, K3) {
    K2(n,m,i,j,k,l)  = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l)
}
'''   

kernel_quadratic =\
'''
def quadratic(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy) -> (K2, K3, K4) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l)
    K4(n,m,i,j,k,l) = sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)) *  K3(n,m,i,j,k,l)
}
'''

kernel_quadratic_project =\
'''
def quadratic_project(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy) -> (K2, K3) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l)
}
'''

kernel_exponential =\
'''
def exponential(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy) -> (K2, K3) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = exp(K2(n,m,i,j,k,l))
}
'''

kernel_exponential_shifted  =\
'''
def exponential_shifted(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy, double gamma) -> (K2, K3) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)) * exp(gamma * (K2(n,m,i,j,k,l) - 1))
}

'''

kernel_exponential_shifted_project  =\
'''
def exponential_shifted_project(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy, double gamma) -> (K2, K3) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = exp(gamma * (K2(n,m,i,j,k,l) - 1))
}
'''
kernel_relu = kernel_relu.replace("<pi>", str(math.pi))
kernel_relu_project = kernel_relu_project.replace("<pi>", str(math.pi))


template_kernel_group_norm =\
'''
def group_norm_<kernel_size>(double(BX,BY,H,W,H,W) K, double(BX,BX,H,W,H,W) Kxx, double(BY,BY,H,W,H,W) Kyy) -> (group_norms_r, group_norms_c, group_output) {
    group_norms_r(n) +=! Kxx(n,n, wx, wy, wx, wy) / (<kernel_size>*<kernel_size>) where wx in 0:<kernel_size>, wy in 0:<kernel_size>
    group_norms_c(m) +=! Kyy(m,m, wx, wy, wx, wy) / (<kernel_size>*<kernel_size>) where wx in 0:<kernel_size>, wy in 0:<kernel_size>
    group_output(n,m,i,j,k,l) = 1 / sqrt(group_norms_r(n)) / sqrt(group_norms_c(m)) * (K(n,m,i,j,k,l))
}
'''

kernel_group_norm_32 = template_kernel_group_norm.replace("<kernel_size>", "32")
kernel_group_norm_16 = template_kernel_group_norm.replace("<kernel_size>", "16")
kernel_group_norm_8 = template_kernel_group_norm.replace("<kernel_size>", "8")
kernel_group_norm_4 = template_kernel_group_norm.replace("<kernel_size>", "4")

'''
================== FLOAT IMPLEMENTATIONS ========================
'''
template_kernel_float_conv =\
'''
def conv<conv_size>_input(float(B,H,W,P) X, float(B,H,W,P) Y) -> (conv_output) {
    conv_output(n,m,i,j,k,l) +=! (X(n,i + wx, j + wy, c) * Y(m,k + wx, l + wy, c))  where wx in 0:<conv_size>, wy in 0:<conv_size>
    }
'''

template_kernel_conv_all_chan_float  =\
'''
def conv<conv_size>_all_chan_input(float(B,H,W,P) X, float(B,H,W,P) Y) -> (conv_output) {
    conv_output(n,m,i,j,k,l) +=! (X(n,i + wx, j + wy, c0) * Y(m,k + wx, l + wy, c1))  where wx in 0:<conv_size>, wy in 0:<conv_size>
    }
'''
kernel_float_conv3_input = template_kernel_float_conv.replace("<conv_size>", "3")
kernel_float_conv6_input = template_kernel_float_conv.replace("<conv_size>", "6")

template_kernel_float_conv_layer =\
'''
def conv<conv_size>_layer(float(BX,BY,H,W,H,W) K, float(BX, BY,H,W,H,W) Kxx, float(BX,BY,H,W,H,W) Kyy) -> (conv_output) {
    conv_output(n,m,i,j,k,l) +=! K(n,m, i + wx, j + wy, k + wx, l + wy)  where wx in 0:<conv_size>, wy in 0:<conv_size>
}
'''

kernel_float_conv3_layer = template_kernel_float_conv_layer.replace("<conv_size>", "3")

kernel_float_conv3_all_chan_input = template_kernel_conv_all_chan_float.replace("<conv_size>", "3")
kernel_float_conv5_layer = template_kernel_float_conv_layer.replace("<conv_size>", "5")
kernel_float_conv6_layer = template_kernel_float_conv_layer.replace("<conv_size>", "6")

template_kernel_float_pool_layer =\
'''
def pool<pool_size>(float(BX,BY,H,W,H,W) K, float(BX, BY,H,W,H,W) Kxx, float(BX,BY,H,W,H,W) Kyy) -> (pool_output) {
    pool_output(n,m,i,j,k,l) +=!  K(n,m, <pool_stride>*i + wx, <pool_stride>*j + wy, <pool_stride>*k + wz, <pool_stride>*l + wl)  where wx in 0:<pool_size>, wy in 0:<pool_size>, wz in 0:<pool_size>, wl in 0:<pool_size>
    pool_output(n,m,i,j,k,l)  = pool_output(n,m,i,j,k,l)
}
'''

kernel_float_pool2  = template_kernel_float_pool_layer.replace("<pool_size>", "2").replace("<pool_stride>", "2")
kernel_float_pool3  = template_kernel_float_pool_layer.replace("<pool_size>", "3").replace("<pool_stride>", "3")
kernel_float_pool4  = template_kernel_float_pool_layer.replace("<pool_size>", "4").replace("<pool_stride>", "4")
kernel_float_pool8 = template_kernel_float_pool_layer.replace("<pool_size>", "8").replace("<pool_stride>", "8")
kernel_float_pool7 = template_kernel_float_pool_layer.replace("<pool_size>", "7").replace("<pool_stride>", "7")
kernel_float_pool32  = template_kernel_float_pool_layer.replace("<pool_size>", "32").replace("<pool_stride>", "32")
kernel_float_pool30 = template_kernel_float_pool_layer.replace("<pool_size>", "30").replace("<pool_stride>", "30")
kernel_float_pool32 = template_kernel_float_pool_layer.replace("<pool_size>", "32").replace("<pool_stride>", "32")


kernel_float_relu =\
'''
def relu(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy) -> (conv_output_scaled, conv_output_scaled_sq, conv_relu_output) {
    conv_output_scaled(n,m,i,j,k,l)  = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    conv_output_scaled_sq(n,m,i,j,k,l) = conv_output_scaled(n,m,i,j,k,l) * conv_output_scaled(n,m,i,j,k,l)
    conv_relu_output(n,m,i,j,k,l) = sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)) * (1/<pi> * sqrt(1 - conv_output_scaled_sq(n,m,i,j,k,l)) + conv_output_scaled(n,m,i,j,k,l)* (1 - (1/<pi>) * acos(conv_output_scaled(n,m,i,j,k,l))))
}
'''

kernel_float_relu_project =\
'''
def relu_project(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy) -> (conv_output_scaled, conv_output_scaled_sq, conv_relu_output) {
    conv_output_scaled(n,m,i,j,k,l)  = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    conv_output_scaled_sq(n,m,i,j,k,l) = conv_output_scaled(n,m,i,j,k,l) * conv_output_scaled(n,m,i,j,k,l)
    conv_relu_output(n,m,i,j,k,l) = (1/<pi> * sqrt(1 - conv_output_scaled_sq(n,m,i,j,k,l)) + conv_output_scaled(n,m,i,j,k,l)* (1 - (1/<pi>) * acos(conv_output_scaled(n,m,i,j,k,l))))
}
'''

kernel_float_quartic =\
'''
def quartic(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy) -> (K2, K3, K4) {
    K2(n,m,i,j,k,l)  = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l)
    K4(n,m,i,j,k,l) = sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)) *  K3(n,m,i,j,k,l)
}
'''

kernel_float_quartic_project =\
'''
def quartic_project(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy) -> (K2, K3) {
    K2(n,m,i,j,k,l)  = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l)
}
'''   

kernel_float_quadratic =\
'''
def quadratic(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy) -> (K2, K3, K4) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l)
    K4(n,m,i,j,k,l) = sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)) *  K3(n,m,i,j,k,l)
}
'''

kernel_float_quadratic_project =\
'''
def quadratic_project(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy) -> (K2, K3) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = K2(n,m,i,j,k,l) * K2(n,m,i,j,k,l)
}
'''

kernel_float_exponential =\
'''
def exponential(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy) -> (K2, K3) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = exp(K2(n,m,i,j,k,l))
}
'''

kernel_float_exponential_shifted  =\
'''
def exponential_shifted(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy, float gamma) -> (K2, K3) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)) * exp(gamma * (K2(n,m,i,j,k,l) - 1))
}

'''

kernel_float_exponential_shifted_project  =\
'''
def exponential_shifted_project(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy, float gamma) -> (K2, K3) {
    K2(n,m,i,j,k,l) = min(max(K(n,m,i,j,k,l)/ sqrt(Kxx(n,n,i,j,i,j) * Kyy(m,m,k,l,k,l)), -1), 1)
    K3(n,m,i,j,k,l) = exp(gamma * (K2(n,m,i,j,k,l) - 1))
}
'''
kernel_float_relu = kernel_float_relu.replace("<pi>", str(math.pi))
kernel_float_relu_project = kernel_float_relu_project.replace("<pi>", str(math.pi))


template_kernel_float_group_norm =\
'''
def group_norm_<kernel_float_size>(float(BX,BY,H,W,H,W) K, float(BX,BX,H,W,H,W) Kxx, float(BY,BY,H,W,H,W) Kyy) -> (group_norms_r, group_norms_c, group_output) {
    group_norms_r(n) +=! Kxx(n,n, wx, wy, wx, wy) / (<kernel_float_size>*<kernel_float_size>) where wx in 0:<kernel_float_size>, wy in 0:<kernel_float_size>
    group_norms_c(m) +=! Kyy(m,m, wx, wy, wx, wy) / (<kernel_float_size>*<kernel_float_size>) where wx in 0:<kernel_float_size>, wy in 0:<kernel_float_size>
    group_output(n,m,i,j,k,l) = 1 / sqrt(group_norms_r(n)) / sqrt(group_norms_c(m)) * (K(n,m,i,j,k,l))
}
'''

kernel_float_group_norm_32 = template_kernel_float_group_norm.replace("<kernel_float_size>", "32")
kernel_float_group_norm_16 = template_kernel_float_group_norm.replace("<kernel_float_size>", "16")
kernel_float_group_norm_8 = template_kernel_float_group_norm.replace("<kernel_float_size>", "8")
kernel_float_group_norm_4 = template_kernel_float_group_norm.replace("<kernel_float_size>", "4")

all_kernel_strs = {}
all_data = globals()
for k,v in all_data.copy().items():
    if k.startswith("kernel"):
        all_kernel_strs[k] = v

def avg_pool2d_dim(input, kernel_size, dims=-1, **avgpool_args):
    """
    Perform 2D average pool on specified dimensions.
    Args:
        input (arbitrary shape tensor)
        kernel_size (int or tuple)
        dims (tuple): Dimensions to average pool over in input.
    >>> X = torch.rand(5, 3, 224, 224).cuda()
    >>> assert torch.allclose(
    ...     avg_pool2d_dim(X, 3, dims=(2, 3)), F.avg_pool2d(X, 3))
    """
    X = input
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    assert len(dims) == 2
    # Move the specified dimensions to the end.
    # E.g., if X is a 4-d tensor, and dim=(1, 2), then permutation will be
    #   [0, 3, 1, 2]
    d = X.dim()
    dims = sorted((dim if dim != -1 else d - 1) for dim in dims)
    permutation = [i for i in range(d) if i not in dims] + dims
    # In the above case, undo permutation will be
    #   [0, 2, 3, 1]
    undo_permutation = sorted(range(d), key=lambda x: permutation[x])
    X = X.permute(*permutation).contiguous()
    permuted_shape = X.shape
    X = X.view(1, -1, X.shape[-2], X.shape[-1])
    X = F.avg_pool2d(X, kernel_size, **avgpool_args)
    X = X.view(permuted_shape[:-2] + X.shape[-2:])
    X = X.permute(*undo_permutation)
    return X


def avg_poolkd_from_2d(X, kernel_sizes, dims, **avgpool2d_args):
    """
    Performs a KD average pool (where K is even) using 2D average pools.
    Args:
        X (torch.Tensor): Arbitrary shape
    >>> X = torch.rand(5, 3, 10, 10, 10, 10)
    >>> if torch.cuda.is_available():
    ...     X = X.cuda()
    >>> assert torch.allclose(
    ...     avg_poolkd_from_2d(X, 3, dims=(2, 3, 4, 5)),
    ...     avg_poolkd_from_1d(X, 3, dims=(2, 3, 4, 5)))
    """
    assert len(dims) % 2 == 0
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * len(dims)
    dims = sorted(dims)

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * len(dims)
    for i in range(int(len(dims) / 2)):
        dim_pair = (dims[i * 2], dims[i * 2 + 1])
        ks = (kernel_sizes[i * 2], kernel_sizes[i * 2 + 1])
        X = avg_pool2d_dim(X, ks, dim_pair, **avgpool2d_args)
    return X

class TCWrapper(object):
    def __init__(self, tc_cache=None, float32=False):
        self.float32 = float32
        if tc_cache is None:
            opts = tc.make_naive_options_factory()
        else:
            self.cache_loc = tc_cache
            opts = tc.make_load_from_cache_options_factory(cache_filename=tc_cache)
        self.kernel_map = {}
        for k in all_kernel_strs:
            self.kernel_map[k] = tc.define(all_kernel_strs[k],  opts)
    def cast(self, x):
        if self.float32:
            return x.float()
        else:
            return x.double()

    def kernel_map_get(self, kernel_name):
        if self.float32:
            kernel_name = kernel_name.replace("kernel", "kernel_float")
        return self.kernel_map[kernel_name]

    def conv3_input(self, x, y, **kwargs):
        x = self.cast(x)
        y = self.cast(y)
        return self.kernel_map_get("kernel_conv3_input").conv3_input(x,y)/(3*3)

    def conv3zp_all_chan_input(self, x, y, **kwargs):
        x = self.cast(F.pad(x, (0,0,1,1,1,1)))
        y = self.cast(F.pad(y, (0,0,1,1,1,1)))
        return self.kernel_map_get("kernel_conv3_all_chan_input").conv3_all_chan_input(x,y)/(3*3)

    def input(self, x, y, **kwargs):
        x = self.cast(x)
        y = self.cast(y)
        #return self.kernel_map_get("kernel_input").input(x,y)
        N0,H,W,C = x.shape
        N1,H,W,C = y.shape
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        #N0,H,W,C, C, 
        out = torch.mm(x, torch.t(y))
        out = out.reshape(N0, H, W, N1, H, W)
        return out.permute(0,3,1,2,4,5).contiguous()


    def conv3zp_input(self, x, y, **kwargs):
        x = self.cast(F.pad(x, (0,0,1,1,1,1)))
        y = self.cast(F.pad(y, (0,0,1,1,1,1)))
        res = self.kernel_map_get("kernel_conv3_input")
        return res.conv3_input(x,y)/(3*3)

    def conv3(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return self.kernel_map_get("kernel_conv3_layer").conv3_layer(k, kxx, kyy)/(3*3)

    def conv3zp(self, k, kxx, kyy, **kwargs):
        k = F.pad(k, (1,1,1,1,1,1,1,1))
        kxx = F.pad(kxx, (1,1,1,1,1,1,1,1))
        kyy = F.pad(kyy, (1,1,1,1,1,1,1,1))
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return self.kernel_map_get("kernel_conv3_layer").conv3_layer(k, kxx, kyy)/(3*3)

    def conv3zpinorm_input(self, x, y, **kwargs):
        x = F.pad(x, (0,0,1,1,1,1))
        y = F.pad(y, (0,0,1,1,1,1))
        x = self.cast(x)
        y = self.cast(y)
        res = self.kernel_map_get("kernel_conv3_input")
        K_conv = res.conv3_input(x,y)/(3*3)
        hw0 = K_conv.shape[2]*K_conv.shape[3]
        hw1 = K_conv.shape[4]*K_conv.shape[5]
        row_sum = K_conv.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)/hw0
        col_sum = K_conv.sum(dim=4, keepdim=True).sum(dim=5, keepdim=True)/hw1
        tot_sum = col_sum.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)/hw0
        return K_conv - row_sum - col_sum + tot_sum

    def conv3zp_inorm(self, k, kxx, kyy, **kwargs):
        k = F.pad(k, (1,1,1,1,1,1,1,1))
        kxx = F.pad(kxx, (1,1,1,1,1,1,1,1))
        kyy = F.pad(kyy, (1,1,1,1,1,1,1,1))
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        K_conv = self.kernel_map_get("kernel_conv3_layer").conv3_layer(k, kxx, kyy)/(3*3)
        hw0 = K_conv.shape[2]*K_conv.shape[3]
        hw1 = K_conv.shape[4]*K_conv.shape[5]
        row_sum = K_conv.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)/hw0
        col_sum = K_conv.sum(dim=4, keepdim=True).sum(dim=5, keepdim=True)/hw1
        tot_sum = col_sum.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)/hw0
        return K_conv - row_sum - col_sum + tot_sum

    def conv5(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return self.kernel_map_get("kernel_conv5_layer").conv5_layer(k, kxx, kyy)/(5*5)

    def conv5zp(self, k, kxx, kyy, **kwargs):
        k = F.pad(k, (1,1,1,1,1,1,1,1))
        kxx = F.pad(kxx, (1,1,1,1,1,1,1,1))
        kyy = F.pad(kyy, (1,1,1,1,1,1,1,1))
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return self.kernel_map_get("kernel_conv5_layer").conv5_layer(k, kxx, kyy)/(5*5)

    def relu(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret =  self.kernel_map_get("kernel_relu").relu(k, kxx, kyy)
        return ret[-1]

    def relu_project(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret =  self.kernel_map_get("kernel_relu_project").relu_project(k, kxx, kyy)
        return ret[-1]

    def quartic(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret =  self.kernel_map_get("kernel_quartic").quartic(k, kxx, kyy)
        return ret[-1]

    def quartic_project(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret =  self.kernel_map_get("kernel_quartic_project").quartic_project(k, kxx, kyy)
        return ret[-1]

    def quadratic(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret =  self.kernel_map_get("kernel_quadratic").quadratic(k, kxx, kyy)
        return ret[-1]

    def quadratic_project(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret =  self.kernel_map_get("kernel_quadratic_project").quadratic_project(k, kxx, kyy)
        return ret[-1]

    def exponential(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret = self.kernel_map_get("kernel_exponential").exponential(k, kxx, kyy)
        return ret[-1]

    def exponential_shifted(self, k, kxx, kyy, **kwargs):
        gamma  = kwargs.get("gamma", 1.0)
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        if self.float32:
            gamma = torch.tensor(gamma).float().cuda()
        else:
            gamma = torch.tensor(gamma).double().cuda()
        ret = self.kernel_map_get("kernel_exponential_shifted").exponential_shifted(k, kxx, kyy, gamma)
        return ret[-1]

    def exponential_shifted_project(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        gamma  = kwargs.get("gamma", 1.0)
        if self.float32:
            gamma = torch.tensor(gamma).float().cuda()
        else:
            gamma = torch.tensor(gamma).double().cuda()
        ret = self.kernel_map_get("kernel_exponential_shifted_project").exponential_shifted_project(k, kxx, kyy, gamma)
        return ret[-1]

    def pool2(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return avg_poolkd_from_2d(k, kernel_sizes=2, dims=(2, 3, 4, 5))

    def pool4(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return self.kernel_map_get("kernel_pool4").pool4(k, kxx, kyy)/(pow(4,4))

    def pool8(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return self.kernel_map_get("kernel_pool8").pool8(k, kxx, kyy)/(pow(8,4))

    def pool7(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return self.kernel_map_get("kernel_pool7").pool7(k, kxx, kyy)/(pow(7,4))

    def pool30(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return self.kernel_map_get("kernel_pool30").pool30(k, kxx, kyy)/(pow(30,4))

    def pool32(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        return self.kernel_map_get("kernel_pool32").pool32(k, kxx, kyy)/(pow(32,4))

    def group_norm_32(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret = self.kernel_map_get("kernel_group_norm_32").group_norm_32(k, kxx, kyy)
        return ret[-1]

    def group_norm_16(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret = self.kernel_map_get("kernel_group_norm_16").group_norm_16(k, kxx, kyy)
        return ret[-1]

    def group_norm_8(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret = self.kernel_map_get("kernel_group_norm_8").group_norm_8(k, kxx, kyy)
        return ret[-1]

    def group_norm_4(self, k, kxx, kyy, **kwargs):
        k = self.cast(k)
        kxx = self.cast(kxx)
        kyy = self.cast(kyy)
        ret = self.kernel_map_get("kernel_group_norm_4").group_norm_4(k, kxx, kyy)
        return ret[-1]

    @contextlib.contextmanager
    def precision(self, precision):
        assert precision in {"float32", "float64"}
        old_value = self.float32
        self.float32 = (precision == "float32")
        try:
            yield self
        finally:
            self.float32 = old_value



def main():
    parser = argparse.ArgumentParser("compile + tune + test tensor comp kernels...")
    parser.add_argument("--kernel_name", default=r"kernel_*")
    parser.add_argument("--list", const=True, action="store_const", default=False)
    parser.add_argument("--tune", const=True, action="store_const", default=False)
    parser.add_argument("--exact", const=True, action="store_const", default=False)
    parser.add_argument("--float32", const=True, action="store_const", default=False)
    parser.add_argument("--load_cache", const=True, action="store_const", default=False)
    parser.add_argument("--generations", default=10, type=int)
    parser.add_argument("--cache_filename", default="tc_cache", type=str)
    parser.add_argument("--init", default="naive", type=str)
    parser.add_argument("--threads", default=16, type=int)
    parser.add_argument("--pop_size", default=100, type=int)
    parser.add_argument("--crossover_rate", default=80, type=int)
    parser.add_argument("--mutation_rate", default=7, type=int)
    parser.add_argument("--number_elites", default=10, type=int)
    parser.add_argument("--height", default=32, type=int)
    parser.add_argument("--width", default=32, type=int)
    parser.add_argument("--N", default=8, type=int)
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--num_gpus", default=1, type=int)
    args = parser.parse_args()
    matched_kernels = []
    gpus = ",".join([str(x) for x in range(args.num_gpus)])
    print("devices: ", gpus)
    tuner_config = tc.TunerConfig().threads(args.threads).generations(args.generations).pop_size(args.pop_size).crossover_rate(args.crossover_rate).mutation_rate(args.mutation_rate).number_elites(args.number_elites)

    for k in all_kernel_strs:
        if not args.exact:
            if re.match(re.compile(args.kernel_name), k):
                matched_kernels.append(k)
        else:
            if k == args.kernel_name:
                matched_kernels.append(k)

    if args.list:
        print("Kernels available:")
        for k in matched_kernels:
            print("\t" + k)

    if args.init not in ["naive", "pointwise", "mlp"]:
        assert False

    start_options =  tc.MappingOptions(args.init)

    if args.tune:
        if not args.load_cache:
            opts = tc.make_autotuned_options_factory(starting_options=start_options, cache_filename=args.cache_filename, store_to_cache=True, tuner_config=tuner_config)
        else:
            print("loading from cache...")
            opts = tc.make_autotuned_options_factory(load_from_cache=True, cache_filename=args.cache_filename, store_to_cache=True, tuner_config=tuner_config)
    else:
        if not args.load_cache:
            opts = tc.make_naive_options_factory()
        else:
            opts = tc.make_load_from_cache_options_factory(cache_filename=args.cache_filename)
    kernel_fn_map = {}
    N = args.N
    H = args.height
    W = args.width
    torch.manual_seed(0)
    x = torch.randn(N,H,W,args.channels).double().cuda()
    k_input = torch.randn(N,N,H,W,H,W).double().cuda()
    z = torch.tensor(1.0).double().cuda()
    y = x
    if args.float32:
        x = x.float()
        y = y.float()
        k_input = k_input.float()
        z = z.float()

    for k in matched_kernels:
        print(f"Tuning {k}")
        kernel_fn = tc.define(all_kernel_strs[k],  opts)
        kernel_fn_map[k] = kernel_fn
        if "float" in k:
            k_call =  getattr(kernel_fn, k.replace("kernel_float_", ""))
        else:
            k_call =  getattr(kernel_fn, k.replace("kernel_", ""))

        if "input" in k:
            kxy = k_call(x,y)
            print("output: ", kxy)
        else:
            if "exponential_shifted" in k:
                print("calling exponential shifted")
                kxy = k_call(k_input,k_input,k_input, z)
            else:
                kxy = k_call(k_input,k_input,k_input)


if __name__ == "__main__":
    main()
