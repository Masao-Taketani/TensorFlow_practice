import os
import json


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f

class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if "time" not in kwargs:
            kwargs["time"] = time.time()
        self.f_log = open(make_path(path), "w")
        self.f_log.write(json.dumps(kwargs)+"\n")

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_gradient_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].getshape()])
def convert_gradient_to_tensor(x):
    """force gradient to be a dense tensor
    it's often faster to do dense embedding gradient on GPU than sparse on CPU
    """
    return x

def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu
    return _assign
