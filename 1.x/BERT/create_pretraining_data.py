import collections
import random
import tensorflow as tf
import tokenization_sentencepiece as tokenization


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
					"Input raw text file (or comma-separated list of flies).")

flags.DEFINE_string("output_file", None,
					"Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("model_file", None,
					"The model file that the SentencePiece model was trained on.")

flags.DEFINE_string("vocab_file", None,
					"The vocabulary file that the SentencePiece model was trained on.")

flags.DEFINE_bool("do_lower_case", True,
				  "Wether to lower case the input text. Should be True for uncased "
				  "models and False for cased models")

flags.DEFINE_integer("max_seq_length", 128,
					 "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
					 "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer("dupe_factor", 10,
					 "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("short_seq_prob", 0.1,
				   "Probability of creating sequences which are shorter than the "
					"maximum length.")


