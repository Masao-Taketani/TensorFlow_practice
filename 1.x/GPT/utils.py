import os
import json


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
