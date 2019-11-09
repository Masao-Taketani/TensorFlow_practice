import collections
import copy
import json
import math
import re
import numpy as np
import tensorflow as tf


class BertConfig(object):

	def __init__(self, vocab_size, hidden_dims=768, num_hidden_layers=12, num_attention_heads=12,
		dense_hidden_dims=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
		max_seq_length=512, type_vocab_size=16, initializer_std=0.02):

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