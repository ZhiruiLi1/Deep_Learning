from functools import reduce

import numpy as np
import tensorflow as tf
from preprocess import get_data
from tensorflow.keras import Model
import os

# ensures that we run only on cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize emnbedding_size, batch_size, and any other hyperparameters

        self.vocab_size = vocab_size
        self.embedding_size = 30  # TODO
        self.batch_size = 300 # TODO

        self.optimizer = tf.keras.optimizers.Adam(1e-3) # learning rate is 1e-3
        self.loss = tf.keras.losses.sparse_categorical_crossentropy


        # TODO: initialize embeddings and forward pass weights (weights, biases)
        def make_vars(*dims, initializer=tf.random.normal):
          return tf.Variable(initializer(dims, stddev=.1))
        
        self.E = make_vars(self.vocab_size, self.embedding_size)   # E means embedding matrix 
        self.W1 = make_vars(2 * self.embedding_size, self.vocab_size) # because trigram has two inputs 
        self.b1 = make_vars(self.vocab_size)
        self.W2 = make_vars(self.vocab_size, self.vocab_size)   # weight
        self.b2 = make_vars(self.vocab_size)   # bias

    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network
        (i.e. tf.nn.embedding_lookup)

        :param inputs: word ids of shape (batch_size, 2)
        :return: probabilities: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """

        # TODO: Fill in
        embed_lookup_0 = tf.nn.embedding_lookup(self.E, inputs[:, 0])  # (inputs[:, 0], embed_size)
        embed_lookup_1 = tf.nn.embedding_lookup(self.E, inputs[:, 1])  # (inputs[:, 1], embed_size)
        embed_inputs = tf.concat([embed_lookup_0, embed_lookup_1], 1) # (inputs, 2 * embed_size)
        dense1 = tf.matmul(embed_inputs, self.W1) + self.b1
        # (inputs, 2*embed_size) * (2*embed_size, vocab_size) + vocab_size
        dense1_relu = tf.nn.relu(dense1)
        dense2 = tf.matmul(dense1_relu, self.W2) + self.b2
        # (inputs, vocab_size) * (vocab_size, vocab_size) + vocab_size

        prob = tf.nn.softmax(dense2)

        return prob


        

    def loss_function(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, vocab_size)
        :return: the average loss of the model as a tensor of size 1
        """
        # TODO: Fill in
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy

        return tf.reduce_mean(self.loss(labels, probabilities))  # syntax: y_true, y_predict 


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.
    Remember to shuffle your inputs and labels - ensure that they are shuffled
    in the same order. Also you should batch your input and labels here.

    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """

    # TODO Fill in
    
    # shuffle the inputs and labels
    random_index = tf.random.shuffle(np.array(range(train_inputs.shape[0])))
    train_inputs = tf.gather(train_inputs, random_index)
    train_labels = tf.gather(train_labels, random_index)



    # use batch_size to train the model 
    train_steps = int(train_inputs.shape[0] / model.batch_size)

    for i in range(train_steps):
      start = i * model.batch_size
      end = (i + 1) * model.batch_size
      if (i + 1) * model.batch_size > train_inputs.shape[0]:
        end = train_inputs.shape[0]
      X = train_inputs[start:end]
      Y = train_labels[start:end]
      
      with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = model.loss_function(y_pred, Y)

      gradients = tape.gradient(loss, model.trainable_variables)
      model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test(model, test_inputs, test_labels):
    """
    Runs through all test examples. Test input should be batched here.

    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)
    :returns: perplexity of the test set
    """

    # TODO: Fill in
    # NOTE: Ensure a correct perplexity formula (different from raw loss)

    train_steps = int(test_inputs.shape[0] / model.batch_size)

    total_loss = []
    for i in range(train_steps):
      start = i * model.batch_size
      end = (i + 1) * model.batch_size
      if (i + 1) * model.batch_size > test_inputs.shape[0]:
        end = test_inputs.shape[0]
      X = test_inputs[start:end]
      Y = test_labels[start:end]

      # average cross entropy loss for 1 batch (similar to 1 sentence)
      loss = tf.reduce_mean(model.loss_function(model(X), Y))
      total_loss.append(loss)
      
    perplexity = np.exp(np.mean(total_loss))

    return perplexity
      
    

    


def generate_sentence(word1, word2, length, vocab, model):
    """
    Given initial 2 words, print out predicted sentence of targeted length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """

    # NOTE: This is a deterministic, argmax sentence generation

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    output_string = np.zeros((1, length), dtype=np.int)
    output_string[:, :2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        start = end - 2
        output_string[:, end] = np.argmax(model(output_string[:, start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]

    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess
    train2id, test2id, word2id = get_data("train.txt", "test.txt")

    # TO-DO:  Separate your train and test data into inputs and labels
    first_word_train = np.array(train2id[0:len(train2id)-2]).reshape(-1,1)
    second_word_train = np.array(train2id[1:len(train2id)-1]).reshape(-1,1)
    inputs_train = np.concatenate((first_word_train, second_word_train), 1)
    labels_train = train2id[2:len(train2id)]

    first_word_test = np.array(test2id[0:len(test2id)-2]).reshape(-1,1)
    second_word_test = np.array(test2id[1:len(test2id)-1]).reshape(-1,1)
    inputs_test = np.concatenate((first_word_test, second_word_test), 1)
    labels_test = test2id[2:len(test2id)]

    # TODO: initialize model
    trigram = Model(len(word2id))

    # TODO: Set-up the training step
    train(trigram, inputs_train, labels_train)


    # TODO: Set up the testing steps
    perplexity = test(trigram, inputs_test, labels_test)


    # Print out perplexity
    print(perplexity)

    # BONUS: Try printing out sentences with different starting words

    


if __name__ == "__main__":
    main()
