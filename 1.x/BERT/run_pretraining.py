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
# mode="w+t" means it uses text instead of binary data
bert_config_file = tempfile.NamedTemporaryFile(mode="w+t", encoding="utf-8", suffix=".json")
# temporaly create a json file
bert_config_file.write(json.dumps({k: utils.str_to_value(v) for k, v in config["BERT-CONFIG"].items()}))
# after writing the tmp file, it requires to rewind it to the begining position.
bert_config_file.seek(0)
sys.path.append(os.path.join(CURDIR, os.pardir, "bert"))
import modeling
import optimization


flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
	"bert_config_file", None,
	"The config json file corresponding to the pre-trianed BERT model. \
	This specifies the model architecture.")

flags.DEFINE_string(
	"input_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"output_dir", None,
	"The output directory where the model checkpoint will be written.")

flags.DEFINE_string(
	"init_checkpoint", None,
	"Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
	"max_seq_length", 128,
	"The maximum total input sequence length after SentencePiece tokenization. \
	Sequences longer than this will be truncated, and sequences shorter \
	than this will be padded. Must match data generation.")

flags.DEFINE_integer(
	"max_predictions_per_seq", 20,
	"Maximum number of masked LM predictions per sequence. \
	Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
					 "How often to save the model checkpoint.")
####################[I don't know what this is]#########################
flags.DEFNIE_integer("iterations_per_loop", 1000,
					 "How many steps to make in each estimator call.")
########################################################################
flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
	"tpu_name", None,
	"The Cloud TPU to use for training. This should be either the name \
	used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 \
	url.")

flags.DEFINE_string(
	"tpu_zone", None,
	"[Optional] GCE zone where the Cloud TPU is located in. If not \
	specified, we will attempt to automatically detect the GCE project from \
	metadata.")

flags.DEFINE_string(
	"gcp_project", None,
	"[Optional] Project name for the Cloud TPU-enabled project. If not \
	specified, we will attempt to automatically detect the GCE project from \
	metadata.")

flags.DEFINE_string("master", None, "[Optinal] TensorFlow master URL.")

flags.DEFINE_integer(
	"num_tpu_cores", 8,
	"Only used if 'use_tpu' is True. Total number of TPU cores to use.")


