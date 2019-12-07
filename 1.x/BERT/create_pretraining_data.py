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


class TrainingInstance(object):

	def __init__(self,
				 tokens,
				 segment_ids,
				 masked_lm_positions,
				 masked_lm_labels,
				 is_random_next):

		self.tokens = tokens
		self.segment_ids = segment_ids
		self.is_random_next = is_random_next
		self.masked_lm_positions = masked_lm_positions
		self.masked_lm_labels = masked_lm_labels

	def __str__(self):
		s = ""
		s += "tokens: %s\n" % (" ".join(
			[tokenization.printable_text(x) for x in self.tokens]))
		s += "segment_ids: %s\n" % (" ".join(
			[str(x) for x in self.segment_ids]))
		s += "is_random_next: %s\n" % self.is_random_next
		s += "masked_lm_positions: %s\n" % (" ".join(
			[str(x) for x in self.masked_lm_positions]))
		s += "masked_lm_labels: %s\n" % (" ".join(
			[tokenization.printable_text(x) for x in self.masked_lm_labels]))
		s += "\n"
		return s

	def __repr__(self):
		return self.__str__()


def write_instance_to_example_files(instances,
									tokenizer,
									max_seq_length,
									max_predictions_per_seq,
									output_files):
	# Create TF example files from 'TrainingInstance's'.
	writers = []
	for output_file in output_files:
		# TFRecordWriter: A class to write records to a TFRecords file.
		writers.append(tf.python_io.TFRecordWriter(output_file))

	writer_index = 0
	total_written = 0
	for (inst_index, instance) in enumerate(instances):
		input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
		input_mask = [1] * len(input_ids)
		segment_ids = list(instance.segment_ids)
		assert len(input_ids) <= max_seq_length

		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		masked_lm_positions = list(instance.masked_lm_positions)
		masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
		masked_lm_weights = [1.0] * len(masked_lm_ids)

		while len(masked_lm_positions) < max_predictions_per_seq:
			masked_lm_positions.append(0)
			masked_lm_ids.append(0)
			masked_lm_weights.append(0.0)

		next_sentence_label = 1 if instance.is_random_next else 0

		features = collections.OrderedDict()
		features["input_ids"] = create_int_features(input_ids)
		features["input_mask"] = create_int_features(input_mask)
		features["segment_ids"] = create_int_features(segment_ids)
		features["masked_lm_positions"] = create_int_features(masked_lm_positions)
		features["masked_lm_ids"] = create_int_features(masked_lm_ids)
		features["masked_lm_weights"] = create_float_features(masked_lm_weights)
		features["next_sentence_label"] = create_int_features([next_sentence_label])
		"""データの読み込みを効率的にするには、データをシリアライズし、連続的に読み込めるファイルのセット
		（各ファイルは 100-200MB）に保存することが有効です。データをネットワーク経由で流そうとする場合には、
		特にそうです。また、データの前処理をキャッシングする際にも役立ちます。
		TFRecord 形式は、バイナリレコードのシリーズを保存するための単純な形式です。"""
		tf_example = tf.train.Example(features=tf.train.Features(feature=features))
		
		"""tf.Example 用のデータ型
		基本的には tf.Example は {"string": tf.train.Feature} というマッピングです。

		tf.train.Feature メッセージ型は次の3つの型のうち1つをとることができます。（.proto file を参照）このほかの一般的なデータ型のほとんどは、強制的にこれらのうちの1つにすること可能です。

		tf.train.BytesList (次の型のデータを扱うことが可能)

			string
			byte
		tf.train.FloatList (次の型のデータを扱うことが可能)

			float (float32)
			double (float64)
		tf.train.Int64List (次の型のデータを扱うことが可能)

			bool
			enum
			int32
			uint32
			int64
			uint64"""
		# 主要なメッセージはすべて .SerializeToString を使ってバイナリ文字列にシリアライズすることができます。
		# TFRecordWriter instance writes a string record to the file.
		writers[writer_index].write(tf_example.SerializeToString())

		writer_index = (writer_index + 1) % len(writers)
		total_written += 1

		if inst_index < 20:
			tf.logging.info("*** Example ***")
			tf.logging.info("token: %s" % " ".join(
				[tokenizer.printable_text(x) for x in instance.tokens]))

			for feature_name in features.keys():
				feature = features[feature_name]
				values = []
				if feature.int64_list.value:
					values = feature.int64_list.value
				elif feature.float_list.value:
					values = feature.float_list.value
				tf.logging.info(
					"%s: %s" % (feature_name, " ".join([str(x) for x in values])))

	for writer in writers:
		writer.close()

	tf.logging.info("Wrote %d total instances", total_written)


# if you use functions below, you can convert values into data types which are
# compatible to tf.Example
def create_int_feature(values):
	feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
	return feature


def create_float_feature(values):
	feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
	return feature

# This func is to create 'TrainingInstance's from raw text.
def create_training_instances(input_files,
							  tokenizer,
							  max_seq_length,
							  dupe_factor,
							  short_seq_prob,
							  masked_lm_prob,
							  max_predictions_per_seq,
							  rng):

	""" input_files format
	(1) One sentence per line since those sentences are also used for 
		"next sentence prediction" task.
	(2) Blank lines between docs since it does not want 
		"next sentence prediction" task to predict unrelated
		sentences."""
	all_documents = [[]]

	for input_file in input_files:
		with tf.gfile.GFile(input_file, "r") as reader:
			while True:
				line = tokenization.convert_to_unicode(reader.readline())
				if not line:
					break
				line = line.strip()

				# Empty lines are used as document delimiters
				# if 'blank str' -> False
				if not line:
					all_documents.append([])
				tokens = tokenizer.tokenize(line)
				if tokens:
					all_documents[-1].append(tokens)

	# Remove empty documents
	all_documents = [x for x in all_documents if x]
	rng.shuffle(all_documents)

	vocab_words = list(tokenizer.vocab.keys())
	instances = []
	for _ in range(dupe_factor):
		for document_index in range(len(all_documents)):
			intances.extend(
				create_instances_from_document(
					all_documents, document_index, max_seq_length, short_seq_prob,
					masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

	rng.shuffle(instances)
	return instances


def create_instances_from_documents(all_documents,
									document_index,
									max_seq_length,
									short_seq_prob,
									masked_lm_prob,
									max_predictions_per_seq,
									vocab_words,
									rng):
	# To create 'TrainingInstance's for a single document.
	document = all_documents[document_index]

	# exclude vocabs for [CLS], first and second [SEP]
	max_num_tokens = max_seq_length - 3

	# Since we are padding each sentence to make it the max_seq_length,
	# we regularly don't want short sentences. That's because you will
	# waste the machine computation if you use those short sentences.
	# However, to make the data to be realistic, which means sometimes
	# short sentences are written in real life, we also pick short
	# sentences with the prob of 'short_seq_prob', which is 0.1 by default
	target_seq_length = max_num_tokens
	if rng.random() < short_seq_prob:
		# rng.randoint(a, b): pick a num between a and b
		target_seq_length = rng.randint(2, max_num_tokens)

	instances = []
	current_chunk = []
	current_length = 0
	i = 0
	# loop for each token
	while i < len(document):
		segment = document[i]
		current_chunk.append(segment)
		current_length += len(segment)
		if i == len(document) - 1 or current_length >= target_seq_length:
			if current_chunk:
				first_sentence_end = 1
				if len(current_chunk) >= 2:
					first_sentence_end = rng.randint(1, len(current_chunk) - 1)

				first_tokens = []
				for j in range(first_sentence_end):
					first_tokens.extend(current_chunk[j])

				second_tokens = []
				if_random_next = False
				if len(current_chunk) == 1 or rng.random() < 0.5:
					is_random_next = True
					target_second_length = target_seq_length - len(first_tokens)

					for _ in range(10):
						random_document_index = rng.randint(0, len(all_documents) - 1)
						if random_document_index != document_index:
							break

					random_document = all_documents[random_document_index]
					random_start = rng.randint(0, len(random_dcument) - 1)
					for j in range(random_start, len(random_document)):
						second_tokens.extend(random_document[j])
						if len(second_tokens) >= target_second_length:
							break

					num_unused_segments = len(current_chunk) - first_sentence_end
					i -= num_unused_segments

				else:
					is_random_next = False:
					for j in range(first_sentence_end, len(current_chunk)):
						second_tokens.extend(current_chunk[j])
				truncate_seq_pair(first_tokens, second_tokens, max_num_tokens, rng)

				assert len(first_tokens) >= 1
				assert len(second_tokens) >= 1