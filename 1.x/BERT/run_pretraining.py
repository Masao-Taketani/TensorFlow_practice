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


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
					 num_train_steps, num_warmup_steps, use_tpu,
					 use_one_hot_embeddings):

	def model_fn(features, labels, mode, params):
		tf.logging.info("*** Features ***")
		for name in sorted(features.key()):
			tf.logging.info(" name = %s, shape = %s" % (name, features[name].shape))

		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		masked_lm_positions = features["masked_lm_positions"]
		masked_lm_ids = features["masked_lm_ids"]
		masked_lm_weights = features["masked_lm_weights"]
		next_sentence_labels = features["next_sentence_labels"]

		# Class ModeKeys: Standard names for Estimator model modes.
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)

		model = modeling.BertModel(
			config=bert_config,
			is_training=is_training,
			input_ids=input_ids,
			input_mask=input_mask,
			token_type_ids=segment_ids,
			use_one_hot_embeddings=use_one_hot_embeddings)

		(masked_lm_loss, masked_lm_example_loss,
			masked_lm_log_probs) = get_masked_lm_output(
		bert_config, model.get_sequence_output(), model.get_embedding_table(),
		masked_lm_positions, masked_lm_ids, masked_lm_weights)

		(next_sentence_loss, next_sentence_example_loss,
			next_sentence_log_probs) = get_next_sentence_output(
			bert_config, model.get_pooled_output(), next_sentence_labels)

		total_loss = masked_lm_loss + next_sentence_loss

		tvars = tf.trainable_variables()

		initialized_variable_names = {}
		scaffold_fn = None

		if init_checkpoint:
			(assignment_map,
			initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

			if use_tpu:
				def tpu_scaffold():
					# Values are not loaded immediately, but when the initializer is run
					# (typically by running a tf.compat.v1.global_variables_initializer op).
					# need to refer the reference more.
					# https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/init_from_checkpoint
					tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
					return tf.train.Scaffold()

				scaffold_fn = tpu_scaffold
			else:
					# Values are not loaded immediately, but when the initializer is run
					# (typically by running a tf.compat.v1.global_variables_initializer op).
					# need to refer the reference more.
					# https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/init_from_checkpoint
				tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

		tf.logging.info("**** Trainable Variables ****")
		for var in tvars:
			init_string = ""
			if var.name in initialized_variable_names:
				init_string = ", *INIT_FROM_CKPT"
			tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape,
				init_string)

		output_spec = None
		if mode == tf.estimator.ModeKeys.TRAIN:
			train_op = optimization.create_optimizer(
				total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				loss=total_loss,
				train_op=train_op,
				scaffold_fn=scaffold_fn)

		elif mode == tf.estimator.ModeKeys.EVAL:

			def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
						  masked_lm_weights, next_sentence_example_loss,
						  next_sentence_log_probs, next_sentence_labels):

				masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
												[-1, masked_lm_log_probs.shape[-1]])
				masked_lm_predictions = tf.argmax(
					masked_lm_log_probs, axis=-1, output_type=tf.int32)
				masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
				masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
				masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
				###################[What are the weights?]#####################
				masked_lm_accuracy = tf.metrics.accuracy(
					labels=masked_lm_ids,
					predictions=masked_lm_predictions,
					weights=masked_lm_weights)
				masked_lm_mean_loss = tf.metrics.mean(
					values=masked_lm_example_loss, weights=masked_lm_weights)
				###############################################################

				next_sentence_log_probs = tf.reshape(
					next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
				next_sentence_predictions = tf.argmax(
					next_sentence_log_probs, axis=-1, output_type=tf.int32)
				next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
				next_sentence_accuracy = tf.metrics.accuracy(
					labels=next_sentence_labels, predictions=next_sentence_predictions)
				next_sentence_mean_loss = tf.metrics.mean(
					values=next_sentence_example_loss)

				return {
					"masked_lm_accuracy": masked_lm_accuracy,
					"masked_lm_loss": masked_lm_mean_loss,
					"next_sentence_accuracy": next_sentence_accuracy,
					"next_sentence_loss": next_sentence_loss
				}


			eval_metrics = (metric_fn, [
				masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
				masked_lm_weights, next_sentence_example_loss,
				next_sentence_log_probs, next_sentence_labels])

			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				loss=total_loss,
				eval_metrics=eval_metrics,
				scaffold_fn=scaffold_fn)

		else:
			raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

		return output_spec

	return model_fn


