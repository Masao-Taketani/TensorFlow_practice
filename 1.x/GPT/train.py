import os
import time
import math
import json
import joblib
import random
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from functools import partial
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from datasets import rocstories
from analysis import rocstories as rocstories_analysis
from text_utils import TextEncoder
from utils import encode_dataset, flatten, iter_data, find_trainable_variables,\
convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads,\
make_path

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2/math.pi) * (x+0.044715 * tf.pow(x, 3))))

def swish(x):
    return x * tf.nn.sigmoid(x)

opt_fn = {
    "adam": adam,
}

act_fn = {
    "relu": tf.nn.relu,
    "swish": swish,
    "gelu": gelu
}

lr_schedules = {
    "warmup_cosine": warmup_cosine,
    "warmup_linear": warmup_linear,
    "warmup_constant": warmup_constant,
}

def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u). axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x * g + b
    return x

def norm(x, scope, axis=[-1]):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1 - pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w * b + -1e9 * (1 - b)
    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str)
    parser.add_argument("--dataset", type=str)

    args = parser.parse_args()
    print(args)
    """
    [references]
    globals().update: https://stackoverflow.com/questions/1589968/python-difference-between-global-globals-updatevar
    __dict__: http://coolpythontips.blogspot.com/2015/12/dict.html
    """
    globals().update(args.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
