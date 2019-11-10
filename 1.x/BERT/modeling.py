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
		#################################################
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
					embedding_size=config.hidden_dims,
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


