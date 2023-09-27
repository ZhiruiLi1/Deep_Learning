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

        # TODO: initialize embedding_size, batch_size, and any other hyperparameters

        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 30  # TODO
        self.batch_size = 100  # TODO

        self.rnn_size = 30   ## dimension of inner cell such as hidden state, output state, and cell state for exactly one LSTM block 

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.l = tf.keras.losses.sparse_categorical_crossentropy

        ## RNNs
        self.LSTM = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)

        # return_sequences = True: all hidden states (hidden state of all the time steps)
        # return_state = True: last hidden state + last cell state
        
        # Architecture 
        self.emb = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size) 
        # The model will take as input an integer matrix of size (batch_size, window_size)
        # and the largest integer (i.e. word index) in the input should be no larger than vocab_size

        self.mlp1 = tf.keras.layers.Dense(100, activation = 'relu') 
        self.mlp2 = tf.keras.layers.Dense(self.vocab_size, activation = 'softmax')




    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network
        (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state
        (NOTE 1: If you use an LSTM, the final_state will be the last two RNN outputs,
        NOTE 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU
        """

        # TODO: Fill in
        emb_inputs = self.emb(inputs) # embedding lookup
                                      # (batch_size, window_size, embedding_size)

        # print(f'this is the shape of emb_inputs: {emb_inputs.shape}')  # (200, 20, 30)


        whole_seq_output, final_hidden_state, final_cell_state = self.LSTM(emb_inputs, initial_state = initial_state)
        # inputs should be: (batch_size, timesteps(window_size), feature)
        # whole_seq_output: (batch_size, timesteps(window_size), unit/rnn_size)
        # final_hidden_state, final_cell_state: (batch_size, unit/rnn_size)

        dense = self.mlp1(whole_seq_output)
        prob = self.mlp2(dense)

        return prob, (final_hidden_state, final_cell_state)

    def loss(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the average loss of the model as a tensor of size 1
        """

        # TODO: Fill in
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy

        return tf.reduce_mean(self.l(labels, probabilities))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    # TODO: Fill in


    ## reshape
    num = len(train_inputs) // model.window_size
    train_inputs = np.array(train_inputs[0:num*model.window_size]).reshape(-1, model.window_size)
    # divide train_inputs to individual window_size 
    train_labels = np.array(train_labels[0:num*model.window_size]).reshape(-1, model.window_size)

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
        prob, final_state = model.call(X, initial_state = None)
        total_loss = model.loss(prob, Y)
      
      gradients = tape.gradient(total_loss, model.trainable_variables)
      model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """

    # TODO: Fill in
    # NOTE: Ensure a correct perplexity formula (different from raw loss)


    # reshape 
    num = len(test_inputs) // model.window_size
    test_inputs = np.array(test_inputs[0:num*model.window_size]).reshape(-1, model.window_size)
    # divide train_inputs to individual window_size 
    test_labels = np.array(test_labels[0:num*model.window_size]).reshape(-1, model.window_size)


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
      
      prob, final_state = model.call(X, initial_state = None)
      batch_loss = tf.reduce_mean(model.loss(prob, Y))
      total_loss.append(batch_loss)

    perplexity = np.exp(np.mean(total_loss))

    return perplexity

def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    # NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0, 0, :])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n, p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep,
    # so you want to remove the last element from train_x and test_x.
    # You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.

    # TODO: Separate your train and test data into inputs and labels
    train_data, test_data, word2id = get_data("train.txt", "test.txt")

    inputs_train = train_data[0:len(train_data)-1]
    labels_train = train_data[1:len(train_data)]
    inputs_test = test_data[0:len(test_data)-1]
    labels_test = test_data[1:len(test_data)]
    vocab_size = len(word2id)

    # TODO: initialize model and tensorflow variables
    LSTM = Model(vocab_size)

    # TODO: Set-up the training step
    train(LSTM, inputs_train, labels_train)

    # TODO: Set up the testing steps
    perplexity = test(LSTM, inputs_test, labels_test)

    # Print out perplexity
    print(perplexity)

    # BONUS: Try printing out various sentences with different start words and sample_n parameters


if __name__ == "__main__":
    main()
