from glob import glob
import configparser
import os
import sentencepiece as sp


CURDIR = os.path.dirname(os.path.abspath(__file__))
CONFIGPATH = os.path.join(CURDIR, os.pardir, "config.ini")
config = configparser.ConfigParser()
config.read(CONFIGPATH)

TEXTDIR = config["DATA"]["TEXTDIR"]
PREFIX = config["SENTENCEPIECE"]["PREFIX"]
VOCABSIZE = config["SENTENCEPIECE"]["VOCABSIZE"]
CTLSYMBOLS = config["SENTENCEPIECE"]["CTLSYMBOLS"]


def _get_text_file(text_dir=TEXTDIR):
	# f-strings: you can specify variables into strings by
	# writing f"xxx" or F"xxx" (from Python3.6)
	# {text_dir}/**/*.txt can dig any directories or files
	# having .txt files under {text_dir} directory
	file_list = glob(f"{text_dir}/**/*.txt")
	files = ",".join(file_list)
	return files


def train(prefix=PREFIX, vocab_size=VOCABSIZE, ctl_symbols=CTLSYMBOLS):
	files = _get_text_file()
	command = f"--input={files} --model_prefix={prefix} --vocab_size={vocab_size} "\
	+ "--control_symbols={ctl_symbols}"
	sp.SentencePieceTrainer.Train(command)


def main():
	train()


if __name__ == "__main__":
	main()