import collections
import re
import unicodedata
import sentencepiece as sp
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):

	if not init_checkpoint:
		return

	m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
	if m is None:
		return

	model_name = m.group(1)

	lower_models = [
		"uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
		"multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
	]

	cased_models = [
		"cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
		"multi_cased_L-12_H-768_A-12"
	]

	is_bad_config = False
	if model_name in lower_models and not do_lower_case:
		is_bad_config = True
		actual_flag = "False"
		case_name = "lowercased"
		opposite_flag = "True"

	if model_name in cased_models and do_loswer_case:
		is_bad_config = True
		actual_flag = "True"
		case_name = "cased"
		opposite_flag = False

	if is_bad_config:
		raise ValueError("You passed in '--do_lower_case=%s' with '--init_checkpoint=%s'. "
			"However, '%s' seems to be a %s model, so you should pass in '--do_lower_case=%s' "
			"so that the fine-tuning matches how the model was pre-training. If this error is "
			"wrong, please just comment out this check."
			% (actual_flag, init_checkpoint, model_name, case_name, opposite_flag))

# using python3 is assumed.
def convert_to_unicode(text):
	if isinstance(text, str):
		return text
	elif isinstance(text, bytes):
		# errors 引数は、入力文字列に対しエンコーディングルールに従った変換ができなかったときの対応方法を指定します。
		# この引数に使える値は
		# 'strict' (UnicodeDecodeError を送出する)、
		# 'replace' (REPLACEMENT CHARACTER である U+FFFD を使う)、
		# "'ignore' (結果となる Unicode から単に文字を除く) 、
		# 'backslashreplace' (エスケープシーケンス \xNN を挿入する) です。
		# referred from
		# https://qiita.com/inoory/items/aafe79384dbfcc0802cf
		return text.decode("utf-8", "strict")
	else:
		raise ValueError("Unsupported string type: %s" % (type(text)))


def printable_text(text):

	if isinstance(text, str):
		return text
	elif isinstance(text, bytes):
		return text.decode("utf-8", "strict")
	else:
		raise ValueError("Unsupported string type: %s" % (type(text)))

# loads a vocab file into a dict
def load_vocab(vocab_file):

	vocab = collections.OrderedDict()
	index = 0
	with tf.gfile.GFile(vocab_file, "r") as reader:
		while True:
			token = convert_to_unicode(reader.readline())
			if not token:
				break
			token, _ = token.split("\t")
			token = token.strip()
			vocab[token] = index
			index += 1
	return vocab


def convert_by_vocab(vocab, items, unk_info):

	output = []
	for item in items:
		if item in vocab:
			output.append(vocab[item])
		else:
			output.append(unk_info)
	return output


def convert_tokens_to_ids(vocab, tokens):

	return convert_by_vocab(vocab, tokens, unk_info=0)


def convert_ids_to_tokens(inv_vocab, ids):

	return convert_by_vocab(inv_vocab, ids, unk_info="<unk>")


class FullTokenizer(object):

	def __init__(self, model_file, vocab_file, do_lower_case=True):
		self.tokenizer = SentencePieceTokenizer(model_file, do_lower_case=do_lower_case)
		self.vocab = load_vocab(vocab_file)
		self.inv_vocab = {v: k for k, v in self.vocab.items()}

	def tokenize(self, text):
		split_tokens = self.tokenizer.tokenize(text)
		return split_tokens

	def convert_tokens_to_ids(self, tokens):

		return convert_by_vocab(self.vocab, tokens, unk_info=0)

	def convert_ids_to_tokens(self, ids):

		return convert_by_vocab(self.vocab, ids, unk_info="<unk>")


class SentencePieceTokenizer(object):

	def __init__(self, model_file=None, do_lower_case=True):

		self.tokenizer = sp.SentencePieceProcessor()
		if self.tokenizer.load(model_file):
			print("Loaded a trained SentencePiece model.")
		else:
			print("You have to give a path of trained SentencePiece model.")
			sys.exit(1)
		self.do_lower_case = do_lower_case

	def tokenize(self, text):

		text = convert_to_unicode(text)
		if self.do_lower_case:
			text = text.lower()
		output_tokens = self.tokenizer.EncodeAsPieces(text)
		return output_tokens