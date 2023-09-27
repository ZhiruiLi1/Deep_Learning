import pickle
import numpy as np
import tensorflow as tf
import os

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(file_path, first_class, second_class):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	You will want to first extract only the data that matches the 
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This 
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes)
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = unpickled_file[b'labels']
	# print(f'this is inputs.shape: {inputs.shape}')
	# print(f'this is labels: {labels}')
	# print(f'this is labels.shape: {np.shape(labels)}')

	labels = np.array(labels)
	# print((labels == first_class) | (labels == second_class))
	correct_index = np.nonzero((labels == first_class) | (labels == second_class))
	# print(correct_index)

	inputs = inputs[correct_index]
	# print(f'this is new inputs.shape: {inputs.shape}') # new inputs with only class cat and dog

	inputs = tf.reshape(inputs, (-1, 3, 32 ,32)) # (2000, 3, 32, 32)
	# print(f'this is new inputs.shape: {inputs.shape}')


	inputs = tf.transpose(inputs, perm=[0,2,3,1])
	# print(f'this is new inputs.shape: {inputs.shape}') # (2000, 32, 32, 3)


	labels = labels[correct_index]
	# print(f'this is new labels.shape: {np.shape(labels)}') # (2000,)
	# print(labels)


	labels = np.where(labels == first_class, 0, 1 ) # first_class is 0, second_class is 1
	# print(labels)

	labels = tf.one_hot(labels, depth = 2, dtype = tf.uint8) # one hot encoding the labels
	# print(labels)


	inputs = inputs / 255    # normalize the inputs
	# print(inputs[0].shape)

	labels = tf.cast(labels, tf.float32)
	inputs = tf.cast(inputs, tf.float32)

	return inputs, labels
