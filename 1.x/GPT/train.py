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
    parser.add_argument("--log_dir", type=str, default="log/")
    parser.add_argument("--save_dir", type=str, default="save/")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--submission_dir", type=str, default="submission/")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--analysis", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--n_batch", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--lr_warmup", type=float, default=0.002)
    parser.add_argument("--n_ctx", type=int, default=512)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--embd_pdrop", type=float, default=0.1)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--clf_pdrop", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--vector_l2", action="store_true")
    parser.add_argument("--n_gpu", type=int, default=4)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--afn", type=str, default="gelu")
    parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
    parser.add_argument("--encoder_path", type=str, default="model/encoder_bpe_40000.json")
    parser.add_argument("--bpe_path", type=str, default="model/vocab_40000.bpe")
    parser.add_argument("--n_transfer", type=int, default=12)
    parser.add_argument("--lm_coef", type=float, default=0.5)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--e", type=float, default=1e-8)

    args = parser.parse_args()
    print(args)
    """
    globals().update(args.__dict__) makes args as global variables
    [references]
    globals().update: https://stackoverflow.com/questions/1589968/python-difference-between-global-globals-updatevar
    __dict__: http://coolpythontips.blogspot.com/2015/12/dict.html
    """
    globals().update(args.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    logger = ResultLogger(path=os.path.join(log_dir, "{}.json".format(desc)), **args.__dict__)
    text_encoder = TextEncoder(encoder_path, bpe_path)
    encoder =text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3) = encode_dataset(
                                                                            rocstories(data_dir),
                                                                            encoder=text_encoder
                                                                            )
