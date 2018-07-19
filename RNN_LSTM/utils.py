# -*- coding: UTF-8 -*-

import os
import sys
import argparse
import datetime
import collections

import numpy as np
import tensorflow as tf


# Data Path for Training and Testing
data_path = "/home/houwenxin/Desktop/AIProjects/RNN_LSTM/simple-examples/data"
# Model Path for trained parameters
save_path = "./save"

load_file = "train-checkpoint-69"

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=data_path, help='Data Path for Training and Testing.')
parser.add_argument('--load_file', type=str, default=load_file, help='The path of checkpoint file of model variables saved during training.')

args = parser.parse_args()

# Python 3 and Python 2.x
Py3 = sys.version_info[0] == 3

# Read the words in the file and replace "\n" with <eos>
def read_words(filename):
	with tf.gfile.GFile(filename, "r") as f:
		if Py3:
			return f.read().replace("\n", "<eos>").split()
		else:
			return f.read().decode("utf-8").replace("\n", "<eos>").split()

# Build a mapping between words and its unique integers.
# The integers are decided by the number of times that the words appear.
# The more frequently appearing words have smaller numbers, for expample, the: 0; <unk>(stands for unknown): 1.
def build_vocab(filename):
	data = read_words(filename)
	
	# Count words' appearing time and sort them.
	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	
	# From words to integers
	word_to_id = dict(zip(words, range(len(words))))
	return word_to_id

def file_to_word_ids(filename, word_to_id):
	data = read_words(filename)
	return [word_to_id[word] for word in data if word in word_to_id]

# Load data
def load_data(data_path):
	train_path = os.path.join(data_path, "ptb.train.txt")
	valid_path = os.path.join(data_path, "ptb.valid.txt")
	test_path = os.path.join(data_path, "ptb.test.txt")

	# Build vocabulary dictionary
	word_to_id = build_vocab(train_path)

	train_data = file_to_word_ids(train_path, word_to_id)
	valid_data = file_to_word_ids(valid_path, word_to_id)
	test_data = file_to_word_ids(test_path, word_to_id)

	# Number of words
	vocab_size = len(word_to_id)
	# Make a Reverse Dictionary
	id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

	print(word_to_id)
	print("================")
	print("vocab_size: ", vocab_size)
	print("================")
	print(train_data[:10])
        print("================")
	print(" ".join([id_to_word[x] for x in train_data[:10]]))
	print("================")
	return train_data, valid_data, test_data, vocab_size, id_to_word

# Generate Mini-Batches
def generate_batches(raw_data, batch_size, num_steps):
	# Transfer raw data to tensor.
	raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
	
	data_len = tf.size(raw_data)
	batch_len = data_len // batch_size

	# Convert data shape to [batch_size, batch_len]
	data = tf.reshape(raw_data[0:batch_size * batch_len], [batch_size, batch_len])
	
	epoch_size = (batch_len - 1) // num_steps

	# Use range_input_producer for multithreading data input to accelerate training.
	# More efficient than feed_dict.
	# shuffle = False : FIFO
	i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

	x = data[:, i * num_steps: (i + 1) * num_steps]
	x.set_shape([batch_size, num_steps])
	
	y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
	y.set_shape([batch_size, num_steps])

	return x, y
class Input(object):
	def __init__(self, batch_size, num_steps, data):
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = generate_batches(data, batch_size, num_steps)


if __name__ == "__main__":
	load_data()
