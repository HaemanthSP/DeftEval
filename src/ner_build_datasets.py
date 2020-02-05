from common_imports import *
import glob
from pathlib import Path

import corpus

if __name__ == "__main__":
	data_folder = "Data/deu"


	print('Build vocab words (may take a while)')

	for file, max_iter in zip(["train", "testa", "testb"], [15000, 3000, 3100]):
		dataset = CoNLL2003DataFile(data_folder + "/" + file, max_iter=max_iter)

		filename = path_leaf(file)
		if filename.rfind('.') != -1:
			filename = filename[0:filename.rfind('.')]

		words_file = open(data_folder + "/" + filename + ".words.txt", "w")
		tags_file = open(data_folder + "/" + filename + ".tags.txt", "w")

		for words, tags in dataset:
			words_file.write(" ".join(words) + "\n")
			tags_file.write(" ".join(tags) + "\n")

		words_file.close()
		tags_file.close()
