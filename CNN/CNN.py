# -*- coding: utf-8 -*-
"""HW2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q8UTZFns4_heo8W-tj0TLuF57K_or5x-
"""

from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math
import pickle

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

  inputs = tf.cast(inputs, tf.float32)
  labels = tf.cast(labels, tf.float32)

  #print(inputs.shape[0])
  
  return inputs, labels

unpickle('test')
get_data('test', 3, 5)

unpickle('train')
get_data('train', 3, 5)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)
print((a > 3) & (a < 5))
print(np.transpose(np.nonzero((a > 3) & (a < 5))))
a[1,0]

x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
print(x)
print(np.nonzero(x))
np.transpose(np.nonzero(x))
# np.nonzero returns the index where it is nonzero

a = np.arange(9)
print(a)
np.where(a <= 3, 5, -2) # (condition, what happen when condition is true, what happen when condition is not true)
# np.where returns the index where it is true

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__(**kwargs)

        self.batch_size = 100
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.epoch = 20


        # TODO: Initialize all trainable parameters
        def make_variables(*dims, initializer=tf.random.truncated_normal):   ##  *dims takes all unnamed variables and condenses to dims list
            return tf.Variable(initializer(dims, stddev=.1))

        self.filter1 = make_variables(5, 5,  3, 16)  # (f_height, f_width, input_channels, output_channels)
        self.filter2 = make_variables(5, 5, 16, 20)    
        self.filter3 = make_variables(3, 3, 20, 20)
        self.W1      = make_variables(8*8*20, 200)
        self.b1      = make_variables(200)             
        self.W2      = make_variables(200, 100)
        self.b2      = make_variables(100) 
        self.W3      = make_variables(100, 2)
        self.b3      = make_variables(2)   # binary classification 
        self.b1_CNN = make_variables(16)
        self.b2_CNN = make_variables(20)
        self.b3_CNN = make_variables(20)



    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        # CNN layer 1
        layer1 = tf.nn.conv2d(inputs, filters = self.filter1, strides = [1, 1, 1, 1], padding = 'SAME')
        layer1_with_bias = tf.nn.bias_add(layer1, self.b1_CNN)
        mean1, variance1 = tf.nn.moments(layer1_with_bias, axes=[0, 1, 2])   # axes=[0, 1, 2] to collapse samples, height, width  # tf.nn.moments is used to calculate mean and variance
        layer1_norm = tf.nn.batch_normalization(layer1_with_bias, mean1, variance1, offset = None, scale = None, variance_epsilon=1e-5)   # batch normalization 
        layer1_relu = tf.nn.relu(layer1_norm)
        layer1_pool = tf.nn.max_pool(layer1_relu, 2, 2, padding = 'SAME') # 32x32x3 -> 16x16x16
        # 2 is ksize, 2 is strides
        
        # tf.nn.max_pool(input, ksize, strides, padding, data_format=None, name=None)

        # CNN layer 2
        layer2 = tf.nn.conv2d(layer1_pool, filters = self.filter2, strides = [1, 1, 1, 1], padding = 'SAME')
        layer2_with_bias = tf.nn.bias_add(layer2, self.b2_CNN)
        mean2, variance2 = tf.nn.moments(layer2_with_bias, axes=[0, 1, 2])   
        layer2_norm = tf.nn.batch_normalization(layer2_with_bias, mean2, variance2, offset = None, scale = None, variance_epsilon=1e-5)
        layer2_relu = tf.nn.relu(layer2_norm)
        layer2_pool = tf.nn.max_pool(layer2_relu, 2, 2, padding = 'SAME')  # 16x16x16 -> 8x8x20
        
        # CNN layer 3
        layer3 = tf.nn.conv2d(layer2_pool, filters = self.filter3, strides = [1, 1, 1, 1], padding = 'SAME')
        layer3_with_bias = tf.nn.bias_add(layer3, self.b3_CNN)
        mean3, variance3 = tf.nn.moments(layer3_with_bias, axes=[0, 1, 2])   
        layer3_norm = tf.nn.batch_normalization(layer3_with_bias, mean3, variance3, offset = None, scale = None, variance_epsilon=1e-5)
        layer3_relu = tf.nn.relu(layer3_norm)   # 8*8*20

        # flatten the layer  
        out = tf.reshape(layer3_relu, (-1,8*8*20))

        '''
        print()
        print(f'this is out.shape: {out.shape}')
        print(f'this is W1.shape: {self.W1.shape}')
        '''

        # fully connected layer 
        dense_layer1 = tf.matmul(out, self.W1) + self.b1
        dense_layer1 = tf.nn.dropout(dense_layer1, rate = 0.3)    # randomly sets elements to zero to prevent overfitting
        dense_layer1 = tf.nn.relu(dense_layer1)

        dense_layer2 = tf.matmul(dense_layer1, self.W2) + self.b2
        dense_layer2 = tf.nn.dropout(dense_layer2, rate = 0.3)    # add non-linearity
        dense_layer2 = tf.nn.relu(dense_layer2)                   # add non-linearity

        dense_layer3 = tf.matmul(dense_layer2, self.W3) + self.b3

        logits = dense_layer3

        return logits
        

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        loss_NN = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        loss_NN = tf.reduce_mean(loss_NN)

        return loss_NN 


    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """


        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))


        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))    # returns accuracy score 
        # tf.cast convert boolean values in correct_prediction to float32 values

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    # shuffle inputs and labels
    new_index = np.arange(train_inputs.shape[0])
    new_shuffle_index = tf.random.shuffle(new_index)
    new_train_inputs = tf.gather(train_inputs, new_shuffle_index)
    new_train_labels = tf.gather(train_labels, new_shuffle_index)

    """
    print(f'this is shape of new_train_inputs: {new_train_inputs.shape}')
    print(f'this is shape of new_train_labels: {new_train_labels.shape}')
    this is shape of new_train_inputs: (10000, 32, 32, 3)
    this is shape of new_train_labels: (10000, 2
    """

    train_steps = int(train_inputs.shape[0] / model.batch_size)

    for i in range(train_steps):
      start = i * model.batch_size
      end = (i + 1) * model.batch_size
      if (i + 1) * model.batch_size > train_inputs.shape[0]:
        end = train_inputs.shape[0]
      new_train_inputs_flip = tf.image.random_flip_left_right(new_train_inputs[start:end]) # randomly flip the image inputs
      new_train_labels_flip = new_train_labels[start:end]

      """
      print()
      print(f'this is shape of new_train_inputs_flip: {new_train_inputs_flip.shape}')
      print(f'this is shape of new_train_labels_flip: {new_train_labels_flip.shape}')
      this is shape of new_train_inputs_flip: (100, 32, 32, 3)
      this is shape of new_train_labels_flip: (100, 2)
      """

      with tf.GradientTape() as tape:
        y_pred = model(new_train_inputs_flip)
        loss = model.loss(y_pred, new_train_labels_flip)

      gradients = tape.gradient(loss, model.trainable_variables)
      model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    
    y_pred = model(test_inputs)
    accuracy = model.accuracy(y_pred, test_labels)
    return accuracy
    


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    train_inputs, train_labels = get_data('train', 3, 5)
    test_inputs, test_labels = get_data('test', 3, 5)

    model = Model()

    for i in range(model.epoch):
      train(model, train_inputs, train_labels)
    
    accuracy = test(model, test_inputs, test_labels)
    print(f'the accuracy for the model is: {accuracy}')

    


if __name__ == '__main__':
    main()

x = tf.constant([[1., 1.], [2., 2.]])
print(x)
x = tf.reduce_mean(x)
x.numpy()

a = np.array([[1,2,3],
              [4,5,6],
              [6,7,8],
              [7,8,9]])
print(a)
a[1:3]

a = np.arange(5)
print(a)
b = tf.random.shuffle(a)
b.numpy()

params = tf.constant([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9,10, 11, 12]])
params.numpy()

tf.gather(params, [2, 2, 1, 0]).numpy()

B = tf.constant([[2, 20, 30, 3, 6], [3, 11, 16, 1, 8], [14, 45, 23, 5, 27]])
print(B)
tf.math.argmax(B, 0).numpy() # argmax in columns
tf.math.argmax(B, 1).numpy() # argmax in rows

a = np.array([[1,2,3],
             [4,5,6]])
b = np.array([[7,8,9],
             [4,5,6]])
c = np.array([[1,2],
              [3,4],
              [6,7]])
d = np.multiply(a,b)
print(d)
e = tf.reduce_sum(d)
print(e.numpy())
f = np.dot(a,c)
f

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
	num_examples = inputs.shape[0]
	in_height = inputs.shape[1]
	in_width = inputs.shape[2]
	input_in_channels = inputs.shape[3]

	filter_height = filters.shape[0]
	filter_width = filters.shape[1]
	filter_in_channels = filters.shape[2]
	filter_out_channels = filters.shape[3]

	num_examples_stride = strides[0]
	strideY = strides[1]
	strideX = strides[2]
	channels_stride = strides[3]

	# input's number of in_channels is equivalent to the filters' number of in_channels
	assert input_in_channels == filter_in_channels

	# Cleaning padding input
	if padding == "SAME":
		padY = math.floor((filter_height - 1) / 2)
		padX = math.floor((filter_width - 1) / 2)	 
		# The math.floor() method rounds a number DOWN to the nearest integer, if necessary, and returns the result
	else:
		padY = 0
		padX = 0
	
	
	inputs_pad = np.pad(inputs, [(0, 0), (padY, padX), (padY, padX), (0, 0)], mode = 'constant')
  # first dimension(num_examples)--no paddingl; fourth dimension(in_channels)--no padding
	# second dimension(in_height)--padding(pad_Y, pad_X)  # padding(before, after)  # (padding for left and above, padding for right and below) 
																																									# [()()] first dimension: row; second dimension: column
	# third dimension(in_height)--padding(pad_Y, pad_X)


	# Calculate output dimensions
	output_height = math.floor((in_height + 2 * padY - filter_height) / strideY + 1)  # strideY = 1
	output_width = math.floor((in_width + 2 * padX - filter_width) / strideX + 1)     # strideX = 1
	output_channels = filter_out_channels
	outputs = np.zeros((num_examples, output_height, output_width, output_channels))

	for cha in range(output_channels):
		for width in range(output_width):
			for height in range(output_height):
				for num in range(num_examples):
					out = inputs_pad[num, height:height+filter_height, width:width+filter_width , :]
					outputs[num, height, width, cha] = tf.reduce_sum(tf.multiply(out, filters[:,:,:,cha]))
		 			
					# print(f'this is outputs[0].shape: {outputs[0].shape}')
					# this is outputs[0].shape: (4, 4, 1)

		 			# inputs: [num_examples, in_height, in_width, in_channels]
					# filters: [filter_height, filter_width, in_channels, out_channels]
					# outputs: [num_examples, output_height, output_width, output_channels]

		 			# filters[:,:,:,cha] represents each individual filter 
		 			# np.multiply is the element wise multiply 
					# tf.reduce_sum adds everything together 

	print(f'this is outputs[0].shape: {outputs[0].shape}')
	"""
  this is outputs[0].shape: (4, 4, 1)
	print(f'this is outputs[0]: {outputs[0]}')
	"""

	# PLEASE RETURN A TENSOR. HINT: tf.convert_to_tensor(your_array, dtype = tf.float32)
	outputs = tf.convert_to_tensor(outputs, dtype = tf.float32)
	return outputs
 

def same_test_0():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def main():
	# TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output
	same_test_0()
	valid_test_0()
	valid_test_1()
	valid_test_2()


if __name__ == '__main__':
	main()

a = np.zeros((2,3,4,5))
a