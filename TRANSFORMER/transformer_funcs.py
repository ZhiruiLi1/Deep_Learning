import numpy as np
import tensorflow as tf
import numpy as np

from attenvis import AttentionVis
av = AttentionVis()

@av.att_mat_func
def Attention_Matrix(K, Q, use_mask=False):
	"""
	This functions runs a single attention head.
	:param K: is [batch_size x window_size_keys x embedding_size]
	:param Q: is [batch_size x window_size_queries x embedding_size]
	:return: attention matrix
	"""
	window_size_queries = Q.get_shape()[1] # window size of queries
	window_size_keys = K.get_shape()[1] # window size of keys
	mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
	atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

	# TODO:
	# 1) compute attention weights using queries and key matrices (if use_mask==True, then make sure to add the attention mask before softmax)
	# 2) return the attention matrix

	QK = tf.matmul(Q, K, transpose_b = True)   # transpose_b refers to K 
	QK_scale = tf.divide(QK, np.sqrt(window_size_keys)) # (batch_size x window_size_queries x window_size_keys x batch_size)

	if use_mask:
		QK_scale = QK_scale + atten_mask
	
	# the mask is a series of -infinity and 0s, so when you add, the places that are masked become infinitely small
	# prevent the model from peaking ahead of future words
	
	QK_scale = tf.nn.softmax(QK_scale)

	return QK_scale

	# Check lecture slides for how to compute self-attention.
 	# You can use tf.transpose or tf.tensordot to perform the matrix multiplication for 3D matrices
	# Remember:
	# - Q is [batch_size x window_size_queries x embedding_size]
	# - K is [batch_size x window_size_keys x embedding_size]
	# - Mask is [batch_size x window_size_queries x window_size_keys]


	# Here, queries are matmuled with the transpose of keys to produce for every query vector, weights per key vector.
	# This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
	# Those weights are then used to create linear combinations of the corresponding values for each query.
	# Those queries will become the new embeddings.


class Atten_Head(tf.keras.layers.Layer):
	def __init__(self, input_size, output_size, use_mask):
		super(Atten_Head, self).__init__()

		self.use_mask = use_mask

		# TODO:
		# Initialize the weight matrices for K, V, and Q.
		# They should be able to multiply an input_size vector to produce an output_size vector
		# Hint: use self.add_weight(...)
		self.weight_K = self.add_weight(shape = (input_size, output_size), initializer = 'random_normal', trainable = True)
		self.weight_V = self.add_weight(shape = (input_size, output_size), initializer = 'random_normal', trainable = True)
		self.weight_Q = self.add_weight(shape = (input_size, output_size), initializer = 'random_normal', trainable = True)  
	


	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

		"""
		This functions runs a single attention head.
		:param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]   # param K: is [batch_size x window_size_keys x embedding_size]
		:param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
		"""

		# TODO:
		# - Apply 3 matrices to turn inputs into keys, values, and queries. You will need to use tf.tensordot for this.
		# - Call Attention_Matrix with the keys and queries, and with self.use_mask.
		# - Apply the attention matrix to the values

		K = tf.matmul(inputs_for_keys, self.weight_K)
		V = tf.matmul(inputs_for_values, self.weight_V)
		Q = tf.matmul(inputs_for_queries, self.weight_Q)

		weighting = Attention_Matrix(K, Q, use_mask = self.use_mask)
		sum = tf.matmul(weighting, V)
	
		return sum



class Multi_Headed(tf.keras.layers.Layer):
	def __init__(self, emb_sz, use_mask):
		super(Multi_Headed, self).__init__()

		# TODO:
		# Initialize heads
		self.num_heads = 3  # output_size is smaller for multi_headed attention 
		self.head1 = Atten_Head(emb_sz, emb_sz//self.num_heads, use_mask)
		self.head2 = Atten_Head(emb_sz, emb_sz//self.num_heads, use_mask)
		self.head3 = Atten_Head(emb_sz, emb_sz//self.num_heads, use_mask)
		self.dense = Feed_Forwards(emb_sz)  # output size is emb_sz
																				# go through a linear layer 

	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
		"""
		FOR CS2470 STUDENTS:
		This functions runs a multiheaded attention layer.
		Requirements:
			- Splits data for 3 different heads into size embed_sz/3
			- Create three different attention heads
			- Each attention head should have input size embed_size and output embed_size/3
			- Concatenate the outputs of these heads together
			- Apply a linear layer
		:param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
		"""
		attention_head1 = self.head1(inputs_for_keys, inputs_for_values, inputs_for_queries) # calling the Atten_Head
		attention_head2 = self.head2(inputs_for_keys, inputs_for_values, inputs_for_queries)
		attention_head3 = self.head2(inputs_for_keys, inputs_for_values, inputs_for_queries)
		multi_heads = tf.concat([attention_head1, attention_head2, attention_head3], 2) # concat horizontally 
		multi_heads_after_linear = self.dense(multi_heads)

		return multi_heads_after_linear



class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Feed_Forwards, self).__init__()

		self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')  # output size is emb_sz
		self.layer_2 = tf.keras.layers.Dense(emb_sz)   # output size is emb_sz

	@tf.function
	def call(self, inputs):
		"""
		This functions creates a feed forward network as described in 3.3
		https://arxiv.org/pdf/1706.03762.pdf
		Requirements:
		- Two linear layers with relu between them
		:param inputs: input tensor [batch_size x window_size x embedding_size]
		:return: tensor [batch_size x window_size x embedding_size]
		"""
		layer_1_out = self.layer_1(inputs)
		layer_2_out = self.layer_2(layer_1_out)
		return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
	def __init__(self, emb_sz, is_decoder, multi_headed=False):
		super(Transformer_Block, self).__init__()

		self.ff_layer = Feed_Forwards(emb_sz)
		self.self_atten = Atten_Head(emb_sz,emb_sz,use_mask=is_decoder) if not multi_headed else Multi_Headed(emb_sz,use_mask=is_decoder)
		self.is_decoder = is_decoder
		if self.is_decoder:
			self.self_context_atten = Atten_Head(emb_sz,emb_sz,use_mask=False) if not multi_headed else Multi_Headed(emb_sz,use_mask=False)

		self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

	@tf.function
	def call(self, inputs, context=None):
		"""
		This functions calls a transformer block.
		There are two possibilities for when this function is called.
		    - if self.is_decoder == False, then:
		        1) compute unmasked attention on the inputs
		        2) residual connection and layer normalization
		        3) feed forward layer
		        4) residual connection and layer normalization
		    - if self.is_decoder == True, then:
		        1) compute MASKED attention on the inputs (self-attention)
		        2) residual connection and layer normalization
		        3) computed UNMASKED attention using context (encoder-decoder attention)
		        4) residual connection and layer normalization
		        5) feed forward layer
		        6) residual connection and layer normalization

		If the multi_headed==True, the model uses multiheaded attention (Only 2470 students must implement this)
		:param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
		:context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
			default=None, This is context from the encoder to be used as Keys and Values in self-attention function
		"""

		with av.trans_block(self.is_decoder):
			atten_out = self.self_atten(inputs,inputs,inputs)
		atten_out+=inputs
		atten_normalized = self.layer_norm(atten_out)

		if self.is_decoder:
			assert context is not None,"Decoder blocks require context"
			context_atten_out = self.self_context_atten(context,context,atten_normalized)
			context_atten_out+=atten_normalized
			atten_normalized = self.layer_norm(context_atten_out)

		ff_out=self.ff_layer(atten_normalized)
		ff_out+=atten_normalized
		ff_norm = self.layer_norm(ff_out)

		return tf.nn.relu(ff_norm)

class Position_Encoding_Layer(tf.keras.layers.Layer):
	def __init__(self, window_sz, emb_sz):
		super(Position_Encoding_Layer, self).__init__()
		self.positional_embeddings = self.add_weight("pos_embed",shape=[window_sz, emb_sz])

	@tf.function
	def call(self, x):
		"""
		Adds positional embeddings to word embeddings.
		:param x: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
		:return: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
		"""
		return x+self.positional_embeddings
