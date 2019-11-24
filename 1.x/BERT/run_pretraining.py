import configparser
import json
import os
import sys
# to craete tmp file or directory
import tempfile
import tensorflow as tf
import utils

CURDIR = os.path.dirname(os.path.abspath(__file__))
CONFIGPATH = os.path.join(CURDIR, os.pardir, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)
####################[need to check here]#######################
bert_config_file = tempfile.NamedTemporaryFile(mode="w+t", encoding="utf-8", suffix=".json")
bert_config_file.write(json.dumps({k: utils.str_to_value(v) for k, v in config["BERT-CONFIG"].items()}))
bert_config_file.seek(0)
###############################################################
sys.path.append(os.path.join(CURDIR, os.pardir, "bert"))
import modeling
import optimization