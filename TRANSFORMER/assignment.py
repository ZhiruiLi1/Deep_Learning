import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random

from attenvis import AttentionVis
av = AttentionVis()

def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: French train data (all data for training) of shape (num_sentences, window_size)
	:param train_english: English train data (all data for training) of shape (num_sentences, window_size + 1)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the French sentences to be used by the encoder,
	# and English sentences to be used by the decoder
	# - The English sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP]
	#
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]	

	# shuffling 
	random_indexes = np.arange(train_french.shape[0])
	random_indexes = tf.random.shuffle(random_indexes)
	# encoder
	train_f = tf.gather(train_french, random_indexes)
	# decoder 
	train_e_inputs = tf.gather(train_english[:, :14], random_indexes)  # last token removed
	train_e_labels = tf.gather(train_english[:,1:15], random_indexes)  # first token removed  # still the same size with the train_f (train_french)

	batch_size = model.batch_size
	iterations = train_french.shape[0]//batch_size

	for i in range(iterations):
		batch_f = train_f[i*batch_size:(i+1)*batch_size, :]
		batch_e_inputs = train_e_inputs[i*batch_size:(i+1)*batch_size, :]
		batch_e_labels = train_e_labels[i*batch_size:(i+1)*batch_size, :]
		mask = tf.where(tf.equal(batch_e_labels, eng_padding_index), False, True)  # mask has the same dimension as the labels
		# tf.equal(): returns the truth value of (x == y) element-wise
		# tf.where(): when tf.equal is true, then tf.where will say false (has padding)
		# when tf.equal is false, then tf.where will say true (no padding/real word)

		# calculate loss
		with tf.GradientTape() as tape:
			logits = model.call(batch_f, batch_e_inputs)
			loss = model.loss_function(logits, batch_e_labels, mask)
	 
	  # apply gradients/backpropagation
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@av.test_func
def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: French test data (all data for testing) of shape (num_sentences, window_size)
	:param test_english: English test data (all data for testing) of shape (num_sentences, window_size + 1)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
	e.g. (my_perplexity, my_accuracy)
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!
	batch_size = model.batch_size
	iterations = test_french.shape[0]//batch_size

	# decoder 
	train_e_inputs = test_english[:, :14]  # last token removed
	train_e_labels = test_english[:,1:15]  # first token removed 

	total_loss = []
	accuracy = []
	all_token = 0    # ignore padding, only use real words to calculate perplexity and accuracy 

	for i in range(iterations):
		batch_f = test_french[i*batch_size:(i+1)*batch_size, :]
		batch_e_inputs = train_e_inputs[i*batch_size:(i+1)*batch_size, :]
		batch_e_labels = train_e_labels[i*batch_size:(i+1)*batch_size, :]
		mask = tf.where(tf.equal(batch_e_labels, eng_padding_index), False, True)
		# tf.equal(): returns the truth value of (x == y) element-wise
		# tf.where(): when tf.equal is true, then tf.where will say false (has padding)
		# when tf.equal is false, then tf.where will say true (no padding/it is a real word)

		token = np.sum(mask)
	
		logits = model.call(batch_f, batch_e_inputs)  # calculate the logits
		batch_loss = model.loss_function(logits, batch_e_labels, mask)  # calculate the batch loss
		batch_accuracy = model.accuracy_function(logits, batch_e_labels, mask) # calculate the batch accuracy
		total_loss.append(batch_loss)
		accuracy.append(batch_accuracy*token) # calculate the accuracy 
	
		all_token += token
	
	perplexity = np.exp(sum(total_loss)/all_token)  # total perplexity for the test data 
	average_accuracy = sum(accuracy)/all_token

	return (perplexity, average_accuracy)

def main():

	model_types = {"RNN" : RNN_Seq2Seq, "TRANSFORMER" : Transformer_Seq2Seq}
	if len(sys.argv) != 2 or sys.argv[1] not in model_types.keys():
		print("USAGE: python assignment.py <Model Type>")
		print("<Model Type>: [RNN/TRANSFORMER]")
		exit()

	# Change this to "True" to turn on the attention matrix visualization.
	# You should turn this on once you feel your code is working.
	# Note that it is designed to work with transformers that have single attention heads.
	if sys.argv[1] == "TRANSFORMER":
		av.setup_visualization(enable=False)

	print("Running preprocessing...")
	data_dir   = '../../data'
	file_names = ('fls.txt', 'els.txt', 'flt.txt', 'elt.txt')
	# file_paths = ['fls.txt', 'els.txt', 'flt.txt', 'elt.txt']
	file_paths = [f'{data_dir}/{fname}' for fname in file_names]
	train_eng,test_eng, train_frn,test_frn, vocab_eng,vocab_frn,eng_padding_index = get_data(*file_paths)
	print("Preprocessing complete.")

	model = model_types[sys.argv[1]](FRENCH_WINDOW_SIZE, len(vocab_frn), ENGLISH_WINDOW_SIZE, len(vocab_eng))
	# model = RNN_Seq2Seq(FRENCH_WINDOW_SIZE, len(vocab_frn), ENGLISH_WINDOW_SIZE, len(vocab_eng))
	
	# TODO:
	# Train and Test Model for 1 epoch.
	train(model, train_frn, train_eng, eng_padding_index)
 
	score = test(model1, test_frn, test_eng, eng_padding_index)
	perplexity = score[0]
	accuracy = score[1]


	# Visualize a sample attention matrix from the test set
	# Only takes effect if you enabled visualizations above
	av.show_atten_heatmap()
	pass

if __name__ == '__main__':
	main()
