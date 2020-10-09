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

def _attn(q, k, v, train=False, scale=False):
    w = tf.matmul(q, k)
    if scale:
        n_state = shape_list(v)[-1]
        w = w * tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)
    w = dropout(w, attn_pdrop, train)
    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    """
    from [batch, n_ctx, n_embd]
    to [batch, n_ctx, n_head, n_embd//n_head]
    """
    new_x_shape = x_shape[:-1] + [n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        """
        from [batch, n_ctx, n_head, n_embd//n_head]
        to [batch, n_head, n_embd//n_head, n_ctx]
        """
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        """
        from [batch, n_ctx, n_head, n_embd//n_head]
        to [batch, n_head, n_ctx, n_embd//n_head]
        """
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def conv1d(x,
           scope,
           nf,
           rf,
           w_init=tf.random_normal_initializer(stddev=0.02),
           b_init=tf.constant_initializer(0),
           pad="VALID",
           train=False):

    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        # for faster 1x1 conv
        if rf == 1:
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]),
                                     tf.reshape(w, [-1, nf])) + b,
                           shape_list(x)[:-1]+[nf])
        # was used to train LM
        else:
            c = tf.nn.conv1d(X, w, stride=1, padding=pad) + b
        return c

def attn(x, scope, n_state, n_head, train=False, scale=False):
    assert n_state % n_head == 0
    with tf.variable_scope(scope):
        c = conv1d(x, "c_attn", n_state*3, 1, train=train)
        """
        the blow code splits linear transformed x into 3 apart by the embedding axis
        (e.g.) if n_embd = 768, then the embeddings for q, k and v will be
        768/3 = 256
        """
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, "c_proj", n_state, 1, train=train)
        a = dropout(a, resid_pdrop, train)
        return a

def mlp(x, scope, n_state, train=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fn[afn]
        h = act(conv1d(x, "c_fc", n_state, 1, train=train))
        h2 = conv1d(h, "c_proj", nx, 1, train=train)
        h2 = dropout(h2, resid_ndrop, train)
        return h2

def block(x, scope, train=False, scale=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, "attn", nx, n_head, train=train, scale=scale)
        n = norm(x + a, "ln_1")
        m = mlp(n, "mlp", nx * 4, train=train)
        h = norm(n + m, "ln_2")
        return h

def embed(X, we):
    we = convert_gradient_to_tensor(we)
    """
    tf.gather(params, indices):
    axis: Defaults to the first non-batch dimension.
    (e.g.)
    >>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    >>> tf.gather(a, [1,2])
    <tf.Tensor: id=15, shape=(2, 3), dtype=float32, numpy= array([[4., 5., 6.],
                                                                  [7., 8., 9.]],
                                                                 dtype=float32)>
    [reference]
    https://www.tensorflow.org/api_docs/python/tf/gather
    https://bit.ly/3iG15cu

    shape of we: [n_vocab + n_special + n_ctx, n_embd]
    shape of X: [batch_size * 2(x12 and x13), n_ctx, 2]
    ? shape of e: [batch_size * 2(x12 and x13), n_ctx, 2, n_embd]
    """
    e = tf.gather(we, X)
    """
    ? shape of h: [batch_size * 2(x12 and x13), n_ctx, n_embd]
    """
    h = tf.reduce_sum(e, 2)
    return h

def model(X, M, Y, train=False, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        we = tf.get_variable("we",
                             [n_vocab + n_special + n_ctx, n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        # after reshaping, X.shape is [batch_size * 2(x12 and x13), n_ctx, 2]
        X = tf.reshape(X, [-1, n_ctx, 2])
        # after reshaping, M.shape is [batch_size * 2(x12 and x13), n_ctx]
        M = tf.reshape(M, [-1, n_ctx])

        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, "h%d"%layer, train=train, scale=True)

        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits,
                                                                   labels=tf.reshape(X[:, 1:, 0],
                                                                                     [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1] - 1])
        lm_losses = tf.recude_sum(lm_losses * M[:, 1:], 1) / tf.recude_sum(M[:, 1:], 1)

        clf_h = tf.reshape(h, [-1, n_embd])
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0],
                                                      clf_token),
                                             tf.float32),
                                     1),
                           tf.int32)
        clf_h = tf.gather(clf_h,
                          tf.range(shape_list(X)[0],
                                   dtype=tf.int32) * n_ctx + pool_idx)

        clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1 - clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, n_embd])
        clf_logits = clf(clf_h, 1, train=train)
        clf_logits = tf.reshape(clf_logits, [-1, 2])

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits,
                                                                    labels=Y)
        return clf_logits, clf_losses, lm_losses

def mgpu_train(*xs):
    gpu_ops = []
    gpu_grads = []
    """
    tf.split(value, num_or_size_splits, axis):
        if you have x such as
       x  = [[[   0    1    2    3    4]
              [   5    6    7    8    9]
              [  10   11   12   13   14]
              [  15   16   17   18   19]
              [  20   21   22   23   24]]

             [[   0   10   20   30   40]
              [  50   60   70   80   90]
              [ 100  110  120  130  140]
              [ 150  160  170  180  190]
              [ 200  210  220  230  240]]

             [[   0  100  200  300  400]
              [ 500  600  700  800  900]
              [1000 1100 1200 1300 1400]
              [1500 1600 1700 1800 1900]
              [2000 2100 2200 2300 2400]]]
    then y1, y2, y3 = tf.split(x, 3, 0) is going to be

        y1 = [[[ 0  1  2  3  4]
               [ 5  6  7  8  9]
               [10 11 12 13 14]
               [15 16 17 18 19]
               [20 21 22 23 24]]]

        y2 = [[[  0  10  20  30  40]
               [ 50  60  70  80  90]
               [100 110 120 130 140]
               [150 160 170 180 190]
               [200 210 220 230 240]]]

        y3 = [[[   0  100  200  300  400]
               [ 500  600  700  800  900]
               [1000 1100 1200 1300 1400]
               [1500 1600 1700 1800 1900]
               [2000 2100 2200 2300 2400]]]
    [reference]https://qiita.com/supersaiakujin/items/464cc053418e9a37fa7b#split
    """
    xs = (tf.split(x, n_gpu, 0) for x in xs)
    for i, xs in enumerate(zip(*xs)):
        do_reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, "/gpu:0")), tf.variable_scope(tf.get_variable_scope(),
                                                                      reuse=do_reuse):
            clf_logits, clf_losses, lm_losses = model(*xs, train=True, reuse=do_reuse)


def transform_roc(X1, X2, X3):
    n_batch = len(X1)
    """
    xmb.shape (batch size, x12 and x13, context length, context tokens and position tokens)
    mmb.shape (batch size, x12 and x13, context length)
    """
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder["_start_"]
    delimiter = encoder["_delimiter_"]
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start] + x1[:max_len] + [delimiter] + x2[:max_len] + [clf_token]
        x13 = [start] + x1[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        # mmb is used for loss calculation for each token position so that
        # the loss is calculated only for non-pad tokens
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    # set indexes for each position
    xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


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
                                                                            encoder=text_encoder)
    n_y = 2
    encoder["_start_"] = len(encoder)
    encoder["_delimiter_"] = len(encoder)
    encoder["_classify_"] = len(encoder)
    clf_token = encoder["_classify_"]
    n_special = 3
    max_len = n_ctx // 2 - 2
    """
    set the context length from the longest sequence from train, validation
    and test datasets + 3(special tokens used in finetuning)
    or the context length which is originally set
    """
    n_ctx = min(max([len(x1[:max_len]) + max(len(x2[:max_len]),
                                             len(x3[:max_len]))
                      for x1, x2, x3 in zip(trX1, trX2, trX3)]
                     +
                     [len(x1[:max_len]) + max(len(x2[:max_len]),
                                              len(x3[:max_len]))
                      for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]
                     +
                     [len(x1[:max_len]) + max(len(x2[:max_len]),
                                              len(x3[:max_len]))
                      for x1, x2, x3 in zip(teX1, teX2, teX3)])
                + 3,
                n_ctx)

    trX, trM = transform_roc(trX1, trX2, trX3)
    vaX, vaM = transform_roc(vaX1, vaX2, vaX3)
    if submit:
        teX, teM = transform_roc(teX1, teX2, teX3)

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = n_batch * n_gpu
    n_updates_total = (n_train // n_batch_train) * n_iter

    # n_batch_train = n_batch * n_gpu is used for the batch size
    X_train = tf.placeholder(tf.int32, [n_batch_train, 2, n_ctx, 2])
    M_train = tf.placeholder(if.float32, [n_batch_train, 2, n_ctx])
    X = tf.placeholder(tf.int32, [None, 2, n_ctx, 2])
    M = tf.placeholder(tf.float32, [None, 2, n_ctx])

    Y_train = tf.placeholder(tf.int32, [n_batch_train])
    Y = tf.placeholder(tf.int32, [None])

    train, logits, clf_losses, lm_losses = mgpu_train(X_train, M_train, Y_train)
