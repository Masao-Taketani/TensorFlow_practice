import configparser
import os
# subprocess is a module that can call os command from python
# for each command line, it has to be splitted by space
# and those splitted pieces have to be put into array
# e.g. "ls -l" -> subprocess.call(["ls", "-l"])
import subprocess
import sys
from urllib.request import urlretrieve


CURDIR = os.path.dirname(os.path.apbpath(__file__))
CONFIGPATH = os.path.join(CURDIR, os.pardir, "config.ini")
config = configparser.ConfigParser()
config.read(CONFIGPATH)

FILEURL = config["DATA"]["FILEURL"]
FILEPATH = config["DATA"]["FILEPATH"]
EXTRACTDIR = config["DATA"]["TEXTDIR"]

# reporthook which is used for urlretrieve accepts 3 srgs.
# 1st arg: # of blocks which have been already transferred so far
# 2nd arg: block size which is shown in bite unit
# 3rd arg: total size of the specified size
# referred from here
# "https://blog.delphinus.dev/2010/05/download-progress-in-python.html"
def reporthook(blocknum, blocksize, totalsize):

	readsofar = blocknum * blocksize
	if totalsize > 0:
		percent = readsofar / totalsize * 100.0
		s = "\r%5.1f%% %*d / %d" % (
			percent, len(str(totalsize)), readsofar, totalsize)
		sys.stderr.write(s)
		if readsofar >= totalsize:
			sys.stderr.write("\n")
	else:
		sys.stderr.write("read %d\n" % (readsofar,))


def download():
	urlretrieve(FILEURL, FILEPATH, reporthook)


def extract():
	# execute the following os command
	subprocess.call(["python3",
					 os.path.join(CURDIR, os.pardir, os.pardir,
					 "wikiextractor", "WikiExtractor.py"),
					 FILEPATH, "-o={}".format(EXTRACTDIR)])


def main():
	download()
	extract()


if __name__ == "__main__":
	main()