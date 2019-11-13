import collections
import copy
import json
import math
import re
import numpy as np
import tensorflow as tf


class BertConfig(object):

	def __init__(self,
				 vocab_size,
				 hidden_dims=768,
				 num_hidden_layers=12,
				 num_attention_heads=12,
				 dense_hidden_dims=3072,
				 hidden_act="gelu",
				 hidden_dropout_prob=0.1,
				 attention_probs_dropout_prob=0.1,
				 max_seq_length=512,
				 # vocab size of "token_type_ids"
				 type_vocab_size=16,
				 initializer_std=0.02):

		self.vocab_size = vocab_size
		self.hidden_dims = hidden_dims
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.hidden_act = hidden_act
		self.dense_hidden_dims = dense_hidden_dims
		self.hidden_dropout_probs = hidden_dropout_probs
		self.attention_probs_dropout_prob = attention_probs_dropout_prob
		self.max_seq_length = max_seq_length
		self.type_vocab_size = type_vocab_size
		self.initializer_std = initializer_std

	@classmethod
	def read_dict(cls, dict):
		###########[need to check later]################
		config = BertConfig(vocab_size=None)
		for (key, value) in dict.items():
			config.__dict__[key] = value
		#################################################
		return config

	@classmethod
	def read_json_file(cls, json_file):
		with tf.gfile.GFile(json_file, "r") as gf:
			text = gf.read()
		return cls.read_dict(json.load(text))

	def make_dict(self):
		# deep copy is used for multi-structured data(such as 2d array)
		# completely not to reference the same objects as original data
		output = copy.deepcopy(self.__dict__)
		return output

	def make_json_from_dict(self):
		return json.dumps(self.make_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
	def __init__(self,
				 config,
				 is_training,
				 input_ids,
				 input_mask=None,
				 # "token_type_ids" = segmentation_ids
				 token_type_ids=None,
				 use_one_hot_embeddings=False,
				 scope=None):

		config = copy.deepcopy(config)
		if not is_training:
			config.hidden_dropout_prob = 0.0
			config.attention_probs_dropout_prob = 0.0

		input_shape = get_shape_list(input_ids, expected_rank=2)
		batch_size, seq_length = input_shape

		if input_mask is None:
			input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

		if token_type_ids is None:
			token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

		with tf.variable_scope(scope, default_name="bert"):
			with tf.variable_scope("embeddings"):
				(self.embedding_output, self.embedding_table) = embedding_lookup(
					input_ids=input_ids,
					vocab_size=config.vocab_size,
					embedding_dims=config.hidden_dims,
					initializer_std=config.initializer_std,
					word_embedding_name="word_embeddings",
					use_one_hot_embeddings=use_one_hot_embeddings

				self.embedding_output = embedding_postprocessor(
					input_tensor=self.embedding_output,
					use_token_type=True,
					token_type_ids=token_type_ids,
					token_type_vocab_size=config.type_vocab_size,
					token_type_embedding_name="token_type_embeddings",
					use_position_embeddings=True,
					position_embedding_name="position_embeddings",
					initializer_std=config.initializer_std,
					max_seq_length=config.max_seq_length,
					dropout_prob=config.hidden_dropout_prob)


			with tf.variable_scope("encoder"):
				attention_mask = create_attention_mask_from_input_mask(
					input_ids, input_mask)

				self.all_encoder_layers = transformer_model(
					input_tensor=self.embedding_output,
					attention_mask=attention_mask,
					hidden_dims=config.hidden_dims,
					num_hidden_layers=config.num_hidden_layers,
					num_attention_heads=config.num_attention_heads,
					dense_hidden_dims=config.dense_hidden_dims,
					intermediate_act_fn=get_activation(config.hidden_act),
					hidden_dropout_prob=config.hidden_dropout_prob,
					attention_probs_dropout_prob=config.attention_probs_dropout_prob,
					initializer_std=config.initializer_std,
					do_return_all_layers=True)

			self.sequence_output = self.all_encoder_layers[-1]

			# The pooler converts the encoded sequence tensor of shape
			# from [batch_size, seq_length, hidden_dims]
			# to [batch_size, hidden_dims] by taking just only the first token
			# The pooler is used for classification tasks
			# The encoder model has to be pre-trained
			with tf.variable_scope("pooler"):
				# tf.squeeze decreases the rank of the specified tensor by the axis
				first_token_tensor = tf.squeeze(self.sequence_output[:, 0, :], axis=1)
				self.pooled_output = tf.layers.dense(
					first_token_tensor,
					config.hidden_dims,
					activation=tf.tanh,
					kernel_initializer=create_initializer(config.initializer_std))

	def get_pooled_output(self):
		return self.pooled_output

	def get_encoded_sequence(self):
		return self.sequence_output

	def get_all_encoder_layers(self):
		return self.all_encoder_layers

	def get_embedding_output(self):
		return self.embedding_output

	def get_embedding_table(self):
		return self.embedding_table


def gelu(x):
	return 0.5 * x * (1.0 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def get_activation(activation_type):
	# if the instance of activation_type is not str,
	# then it is assumed to be an activation function
	# so it returns the value.
	if not isinstance(activation_type, str):
		return activation_type

	# if activation_type is "None",
	# then it is assumed to be linear activation.
	if not activation_type:
		return None

	act = activation_type.lower()
	if act == "linear":
		return None
	elif act == "relu":
		return tf.nn.relu
	elif act == "gelu":
		# this is user-defined function
		return gelu
	elif act == "tanh":
		return tf.tanh
	else:
		raise ValueError("Unsupported activation: %s" % act)

###################[need to check later]######################
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
	assignment_map = {}
	initialized_variable_names = {}

	name_to_variable = collections.OrderedDict()
	for var in tvars:
		name = var.name
		# "(".*)" means strings before ":\"
		# "\\d+$" means int numbers after ":\"
		m = re.match("^(.*):\\d+$", name)
		if m is not None:
			name = m.group(1)
		name_to_variable[name] = var

	init_vars = tf.train.list_variables(init_checkpoint)

	assignment_map = collections.OrderedDict()
	for x in init_vars:
		(name, var) = (x[0], x[1])
		if name not in name_to_variable:
			continue
		assignment_map[name] = name
		initialized_variable_names[name] = 1
		initialized_variable_names[name + ":0"] = 1

	return (assignment_map, initialized_variable_names)
###########################################################

def dropout(input_tensor, dropout_prob):
	if dropout_prob is None or dropout_prob == 0.0:
		return input_tensor

	output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
	return output


def layer_norm(input_tensor, name=None):
	return tf.contrib.layers.layer_norm(
		inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
	output_tensor = layer_norm(input_tensor, name)
	output_tensor = dropout(output_tensor, dropout_prob)
	return output_tensor


def create_initializer(initializer_std=0.02):
	return tf.truncated_normal_initializer(stddev=initializer_std)


def embedding_lookup(input_ids,
					 vocab_size,
					 embedding_dims=128,
					 initializer_std=0.02,
					 word_embedding_name="word_embeddings",
					 use_one_hot_embeddings=False):

	if input_ids.shape.ndims == 2:
		input_ids = tf.expand_dims(input_ids, axis=[-1])

	embedding_table = tf.get_variable(
		name=word_embedding_name,
		shape=[vocab_size, embedding_dims],
		initializer=create_initializer(initializer_std))

	# -1 for shape: the rest of elems supposed to be in this dim,
	# which means [-1] implies all of the elems are in this dim,
	# which is flatten vector.
	flat_input_ids = tf.reshape(input_ids, [-1])
	if use_one_hot_embeddings:
		one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
		output = tf.matmul(one_hot_input_ids, embedding_table)
	else:
		output = tf.gather(embedding_table, flat_input_ids)

	input_shape = get_shape_list(input_ids)

	output = tf.reshape(output,
		input_shape[0:-1] + [input_shape[-1] * embedding_dims])
	return (output, embedding_table)


def embedding_postprocessor(input_tensor,
							use_token_type=False,
							token_type_ids=None,
							token_type_vocab_size=16,
							token_type_embedding_name="token_type_embeddings",
							use_position_embeddings=True,
							position_embedding_name="position_embeddings",
							initializer_std=0.02,
							max_seq_length=512,
							dropout_prob=0.1):

	input_shape = get_shape_list(input_tensor, expected_rank=3)
	batch_size, seq_length, width = input_shape

	output = input_tensor

	if use_token_type:
		if token_type_ids is None:
			raise ValueError("'token_type_ids' must be specified if 'use_token_type' is True.")
		token_type_table = tf.get_variable(
			name=token_type_embedding_name,
			# dims: from 16 to 512 if default value is used
			shape=[token_type_vocab_size, width],
			initializer=create_initializer(initializer_std))
		# use one-hot for token type ids
		flat_token_type_ids = tf.reshape(token_type_ids, [-1])
		one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
		token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
		token_type_embeddings = tf.reshape(token_type_embeddings,
			[batch_size, seq_length, width])
		output += token_type_embeddings

		if use_position_embeddings:
			# if "x <= y" is false, it raises "InvalidArgumentError" 
			assert_op = tf.assert_less_equal(seq_length, max_seq_length)
			# it defines dependency of a computational graphs
			with tf.control_dependencies([assert_op]):
				full_position_embeddings = tf.get_variable(
					name=position_embedding_name,
					shape=[max_seq_length, width],
					initializer=create_initializer(initializer_std))

				# tf.slice(input, begin_position, extract_size)
				position_embeddings = tf.slice(full_position_embeddings,
											   [0, 0],
											   [seq_length, -1])
				#num_dims: the number of dims
				num_dims = len(output.shape.as_list())

				position_broadcast_shape = []
				for _ in range(num_dims - 2):
					position_broadcast_shape.append(1)
				# extend: ex [1].extend([seq_length, width]) => [1, seq_length, width]
				position_broadcast_shape.extend([seq_length, width])
				position_embeddings = tf.rehsape(position_embeddings,
												 position_broadcast_shape)
				output += position_embeddings

		output = layer_norm_and_dropout(output, dropout_prob)
		return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
	from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
	batch_size = from_shape[0]
	from_seq_length = from_shape[1]

	to_shape = get_shape_list(to_mask, expected_rank=2)
	to_seq_length = to_shape[1]

	to_mask = tf.cast(
		tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

	broadcast_ones = tf.ones(
		shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

	mask = broadcast_ones * to_mask
	return mask


def attention_layer(from_tensor,
					to_tensor,
					attention_mask=None,
					num_attention_heads=1,
					size_per_head=512,
					query_act=None,
					key_act=None,
					value_act=None,
					attention_probs_dropout_prob=0.0,
					initializer_std=0.02,
					do_return_2d_tensor=False,
					batch_size=None,
					from_seq_length=None,
					to_seq_length=None):

	