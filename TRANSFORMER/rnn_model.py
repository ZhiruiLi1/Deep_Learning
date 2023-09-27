import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
  def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
    ###### DO NOT CHANGE ##############
    super(RNN_Seq2Seq, self).__init__()
    self.french_vocab_size = french_vocab_size # The size of the French vocab
    self.english_vocab_size = english_vocab_size # The size of the English vocab
    self.french_window_size = french_window_size # The French window size
    self.english_window_size = english_window_size # The English window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters
    def make_vars(*dims, initializer=tf.random.normal):
      return tf.Variable(initializer(dims, stddev=.1))

		# Define batch size and optimizer/learning rate
    self.batch_size = 100 # You can change this
    self.embedding_size = 100 # You should change this
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

		# 2) Define embeddings, encoder, decoder, and feed forward layers
    self.E = make_vars(self.english_vocab_size, self.embedding_size)
    self.F = make_vars(self.french_vocab_size, self.embedding_size)
    self.encoder = tf.keras.layers.LSTM(units = 100, return_sequences=True, return_state=True) # rnn_size is 100
    self.decoder = tf.keras.layers.LSTM(units = 100, return_sequences=True, return_state=True) # rnn_size is 100 
    self.dense = tf.keras.layers.Dense(units = self.english_vocab_size, activation = 'softmax') # dense layer 

    
  @tf.function
  def call(self, encoder_input, decoder_input):
    """
		:param encoder_input: batched ids corresponding to French sentences
		:param decoder_input: batched ids corresponding to English sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""

		# TODO:
		# 1) Pass your French sentence embeddings to your encoder
		# 2) Pass your English sentence embeddings, and final state of your encoder, to your decoder
		# 3) Apply dense layer(s) to the decoder out to generate probabilities
    
    # encoder
    input_french = tf.nn.embedding_lookup(self.F, encoder_input)
    whole_seq_output_f, final_hidden_state_f, final_cell_state_f = self.encoder(input_french, initial_state = None)

    # decoder 
    input_english = tf.nn.embedding_lookup(self.E, decoder_input)
    initial_state = [final_hidden_state_f, final_cell_state_f]

    print("this is the length of final_hidden_state_f")
    print(len(final_hidden_state_f))  # length is 100
    print("this is the length of final_cell_state_f")
    print(len(final_cell_state_f))    # length is 100
    print("this is the length of initial state: ")
    print(len(initial_state))         # length is 2 

    whole_seq_output_e, final_hidden_state_e, final_cell_state_e = self.encoder(input_english, initial_state = initial_state)

    # dense layer 
    logits = self.dense(whole_seq_output_e)
    return logits
    
  def accuracy_function(self, prbs, labels, mask):
    """
		DO NOT CHANGE
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""
    decoded_symbols = tf.argmax(input=prbs, axis=2)
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
    return accuracy
    
  def loss_function(self, prbs, labels, mask):
    """
		Calculates the total model cross-entropy loss after one forward pass.
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
    
    return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs), mask)) 
    # (y_true, y_pred)
