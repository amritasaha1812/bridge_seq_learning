import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn,rnn_cell,seq2seq
from tensorflow.python.ops.math_ops import sigmoid
import scipy as sp
from itertools import chain
import os
import time
import datetime

class BridgeCaptionsModel():
	""" test model for bridge caption generation
	"""

	def __init__(self, max_seq_length, embedding_size, batch_size, rnn_size, e_vocab_size, f_vocab_size, image_size, random_seed, activation=sigmoid):
	    """ initialize the parameters of the RNN 

	    :params:
	      num_classes : int
	        number of classes 
	      vocab_size : int  
	        size of the vocabulary     
	      max_seq_length : int     
	        maximum sequence length allowed
	      embedding_size : int 
	        size of the word embeddings 
	      rnn_size : int
	        size of RNN hidden state  
	      num_layers : int
	        number of layers in the RNN 
	      model : str
	        rnn, gru, basic_lstm, or lstm 
	      init_scale_embedding : float
	        random uniform initialization in the range [-init_scale,init_scale] for the embedding layer
	        (default: 1.0)                            
	      init_scale : float
	        random uniform initialization in the range [-init_scale,init_scale]  
	        (default: 0.1)
	      train_embedding_matrix : boolean
	        if False does not train the embedding matrix and keeps it fixed 
	        (default: True)     
	      use_pretrained_embedding_matrix: boolean
	        if True uses the pretrained embedding matrix passed as a placeholder to the inference funtion
	        (default: False)                      
	    """
		
	    self.max_seq_length = max_seq_length
	    self.embedding_size = embedding_size
	    self.batch_size = batch_size
	    self.rnn_size = rnn_size
	    self.e_vocab_size = e_vocab_size
	    self.f_vocab_size = f_vocab_size
	    self.image_size = image_size
	    self.random_seed = random_seed
	    self.activation=activation
	    self.b_out  = tf.Variable(tf.constant(0.1, shape=[self.f_vocab_size]), name='b_out')

	    """
	    # ==================================================
	    # DECLARE ALL THE PARAMETERS OF THE MODEL
	    # ==================================================
	    """
	    #after training this will contain the word embeddings for the words in e
	    max_val = np.sqrt(6. / (self.e_vocab_size + self.embedding_size))
	    self.W_emb_e = tf.Variable(tf.random_uniform([self.e_vocab_size, self.embedding_size], -1.* max_val, max_val, seed=None, dtype=tf.float32), name="W_emb_en")
	    
	    #after training this will contain the word embeddings for the words in f
	    max_val = np.sqrt(6. / (self.f_vocab_size + self.embedding_size))
	    self.W_emb_f = tf.Variable(tf.random_uniform([self.f_vocab_size, self.embedding_size], -1.* max_val, max_val, seed=None, dtype=tf.float32), name="W_emb_fr")
	    
	    #after training this will contain the word embeddings for the words in f
	    max_val = np.sqrt(6. / (self.image_size + self.embedding_size))
	    self.W_emb_im = tf.Variable(tf.random_uniform([self.image_size, self.embedding_size], -1.* max_val, max_val, seed=None, dtype=tf.float32), name="W_emb_im")
	    #the encoder bias is common for image, en and fr
	    self.b_enc_common = tf.Variable(tf.constant(0., shape=[self.embedding_size]), name='b_enc_common')
	    #the decoder biases are different
	    #check: reset to 0
	    self.b_dec_e = tf.Variable(tf.constant(0.5, shape=[self.e_vocab_size]), name='b_dec_e')
	    self.b_dec_f = tf.Variable(tf.constant(0.5, shape=[self.f_vocab_size]), name='b_dec_f')
	    self.b_dec_im = tf.Variable(tf.constant(0.5, shape=[self.image_size]), name='b_dec_im')

	    # this is the cell which will be used for decoding fr sequences
	    with tf.variable_scope("root", initializer=tf.constant_initializer(0.5), reuse=True):
	    	self.f_cell = rnn_cell.GRUCell(self.rnn_size) # rnn_size is the num_units for GRU

	    max_val = np.sqrt(6. / (self.f_vocab_size + self.rnn_size))
	    self.W_f_rnn_proj = tf.Variable(tf.random_uniform([self.rnn_size, self.f_vocab_size], -1.* max_val, max_val, seed=None, dtype=tf.float32), name="W_f_rnn_proj")
	    self.b_f_rnn_proj = tf.Variable(tf.constant(0., shape=[self.f_vocab_size]), name='b_f_rnn_proj')


	#def output(self):
	#   return self.predictions

	def get_n_hot_rep(self, x, vocab_size):
		"""
		:params:
		  x : tensor, int32 - [batch_size, max_seq_length]
		  each row in the tensor corresponds to one sentence and contains the ids of the words in the sentence (followed by <PAD>s to match
		  	the max_seq_length)
		  vocab_size : 	int 
		  the vocabulary size which will decide the size of the one hot vector
		:return: tensor, int32 - [batch_size, max_seq_length]
		a tensor containing the n-hot encoding of each sentence in x
		"""
		batchsize = x.shape[0]
		n_hot_enc = np.arange(batchsize)*vocab_size
		x_idx = map(np.add, n_hot_enc, x)
		vec = sp.array(list(chain.from_iterable(x_idx)))
		n_hot_enc_W = np.zeros(shape=(batchsize,vocab_size),dtype='int64').flatten()
		n_hot_enc_W[vec] = 1
		n_hot_enc_W = n_hot_enc_W.reshape([batchsize,vocab_size]).astype('int64')
		n_hot_enc_W[:,0:2] = 0 #ignore the <PAD> and <GO> which have word_ids 0 and 1
		return n_hot_enc_W

	def compute_squared_error_loss (self, x, y) :
		"""
		x : tensor, float32 [None, None]
		y : tensor, float32 [None, None] (same shape as x)
		:return: tensor, float32 [None, None] (same shape as x)
		element-wise (x - y)^2 
		"""
		#return tf.square(tf.sub(x, y))
		return tf.reduce_sum(tf.square(tf.sub(x, y))) / self.batch_size

	def compute_hidden_representation(self, sequences, seq_masks, W) :
		""" 
		:params:
		  sequences : tensor, int32 - [batch_size, max_seq_length]
		    indices of the words in the English sentences
		    if the English sentence is "I am at home" and max_seq_length is 8 then the sentence effectively becomes
		    "<GO> I am at home <EOS> <PAD> <PAD>". e_sequences will then be an array of the integer ids of these words 
		    "[id(<GO>),id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]"
		    The same process is repeated for all sentences in the batch	to get a tensor of size [batch_size, max_seq_length]
		  seq_masks : tensor, int32 - [batch_size, max_seq_length]
		    each entry in this tensor indicates whether that word should be ignored or not. For example, for computing h_en
		    we simply sum up the embeddings of all words in the sentence except <PAD> and <GO>. Thus for the above example of  
		    "<GO> I am at home <EOS> <PAD> <PAD>" the seq_masks will be [0,1,1,1,1,1,0,0].
			The same process is repeated for all sentences in the batch	to get a tensor of size [batch_size, max_seq_length]
		  W: tensor, float32 - [vocab_size, embedding_size] or [image_size, embedding_size]
		  	the projection matrix (W_emb_en or W_emb_fr or W_emb_im)
		:return: tensor, int32 - [batch_size, embedding_size]
		  h : the computed hidden representation of the input  	
		"""
		# the next line will return a list of tensors where each tensor corresponds
		# to one sentence and is of dimension (max_seq_length * embedding_size)
		# the length of the list is same as self.batch_size
		sequences = tf.split(0, self.batch_size, tf.nn.embedding_lookup(W, sequences))
		#the above line somehow produces one extra redundant dimension which can be squuezed
		#specifically, instead of producing a list of (max_seq_length, embedding_size) tensors
		#the above line produces a list of (1, max_seq_length, embedding_size) tensors
	 	sequences = [tf.squeeze(sequence_, [0]) for sequence_ in sequences]
	 	
	 	#converts the tensor of size  (batch_size * max_seq_length) into a list of
	 	#1D tensors of size (max_seq_length). The length of the list is same as self.batch_size
		seq_masks = tf.unpack(seq_masks)

		#the representation of a sentence is calculated as the sum of the representations of all words in it plus the bias.
		#However, we want to ignore the <PAD> words at the end. So we first multiply the sentence representation matrix by  
		#the sentence mask vector and add the bias to this product
		# for example if the length of the sentence is 2 and max_seq_length is 4 then the mask vector will be [1,1,0,0]
		#[1,1,0,0] *[1,0] <-- (embeddding of 1st word in the sentence)
		#			[2,2] <-- (embeddding of 2nd word in the sentence)
		#			[3,3] <-- (embeddding of <PAD>)
		#			[3,3] <-- (embeddding of <PAD>)
		# = [3,2] (then add the bias to this)
		# h is a tensor of size (batch_size, embedding_size)
		#there should be a way to optimize the next two lines
		h = [tf.nn.xw_plus_b(tf.reshape(seq_masks[i], (1,self.max_seq_length)), sequences[i], self.b_enc_common) for i in range(len(sequences))] 
		h = tf.squeeze(tf.pack(h), [1]) #the reason for squeezing is same as mentioned above

		return self.activation(h)

	def reconstruct_bow_from_h(self, h, W_dec, b_dec) :
		""" 
		:params:
		  h : tensor, float32 - [batch_size, embedding_size]
		  the hidden representation of en or fr or image
		  W_dec : tensor, float32 - [embedding_size, vocab_size] or [embedding_size, image_size]
		  the decoder matrix (W_dec_en or W_dec_fr or W_dec_im)
		  b_dec : tensor, float32 - [vocab_size] or [image_size]
		  the decoder bias (b_dec_en or b_dec_fr or b_dec_im)	
		:return: tensor, int32 - [batch_size, vocab_size] or [embedding_size, image_size]
		  the input' reconstructed from the hidden representation 	
		""" 
		return self.activation(tf.nn.xw_plus_b(h, W_dec, b_dec))

	def decode_f_seq_from_h(self, f_sequences_to_be_decoded, enc_state, f_cell, 
		f_vocab_size, W_f_rnn_proj, b_f_rnn_proj, feed_previous=False) :
		""" 
		:params:
		  f_sequences_to_be_decoded : list of tensors, float32 - max_seq_length sized list of [batch_size, embedding_size]
		  a list of tensors where each tensor is of size (batch_size). The length of the list is same as max_seq_length
		  this is the format accepted by seq2seq.embedding_rnn_decoder
		  
		  enc_state : tensor, float32 - [batch_size, embedding_size] 
		  this is the state of the encoder computed by the bag-of-words en or fr sentences

		  f_cell : the rnn_cell for decoding French

		  f_vocab_size : int
		  the size of the fr vocabulary

		  W_f_rnn_proj : tensor, float32 - [self.f_rnn_size, self.f_vocab_size]
		  b_f_rnn_proj : tensor, float32 - [self.f_vocab_size]

		  the pair (W_f_rnn_proj, b_f_rnn_proj) of output projection weights and biases; W has shape [f_rnn_size x f_vocab_size] and b has shape [f_vocab_size];
		  if provided and feed_previous=True, each fed previous output will first be multiplied by W and added b.

		  feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of decoder_inputs will be used (the "GO" symbol),
		  and all other decoder inputs will be taken from previous outputs (this will be uset at test time). If False, decoder_inputs are 
		  used as given (this will be used as train time). Setting feed_previous will make this method serve as a greedy decoder
		:return:
		  logits: List of 2D Tensors of shape [batch_size x f_vocab_size]. the length of the list is equal to max_seq_length
		  The i-th row of the j-th tensor in the list contains the logit values calculated for the j-th word in the i-th sentence in the batch [go figure!!]  
		""" 
		#for GRUCell rnn_ouputs and mem_states are the same and essentially contain the hidden representation computed by RNN at each time step
		#it is a list of tensors where each tensor is of the size (self.batch_size, self.rnn_size). The length of the list is equal 
		#to max_seq_lenth. The i-th tensor in the list thus contains the hiddent representation computed at time-step i for every sequence 
		#in the batch
		#seq2seq.embedding_rnn_decoder is a simple rnn decoder which will try to decode "f_sequences_to_be_decoded" taking enc_state as h_zero
		self.rnn_ouputs, mem_states = seq2seq.embedding_rnn_decoder(f_sequences_to_be_decoded, enc_state, 
			f_cell, f_vocab_size, output_projection=(W_f_rnn_proj, b_f_rnn_proj), feed_previous=feed_previous)#
			

		#we now need to take the hidden states (h or rnn_outputs above) and pass them through a feeedforward layer [sig(W_f_rnn_proj*h + b_f_rnn_proj)] 
		#where W_f_rnn_proj is of dimension (f_rnn_size, f_vocab_size). We will then take a softmax over the result to get the Probability distribution 
		#over the vocabulary. 
		
		# Step 1: reshape from (self.max_seq_length, self.batch_size, self.rnn_size) to (self.max_seq_length*self.batch_size, self.rnn_size)
		self.rnn_ouputs = tf.reshape(tf.pack(self.rnn_ouputs), [self.max_seq_length*self.batch_size, self.rnn_size])			
		# Step 2: multiply with W_out [which is also called the output projection matrix] and add b_out
		logits = tf.nn.xw_plus_b(self.rnn_ouputs, W_f_rnn_proj, b_f_rnn_proj, name='logits') 
		predictions = tf.nn.softmax(logits, name='predictions')
		# Step 3: reshape again to (self.max_seq_length, self.batch_size, "self.f_vocab_size") and unpack into a list of tensors as 
		#required later by seq2seq.sequence_loss_by_example()
		logits = tf.unpack(tf.reshape(logits, [self.max_seq_length, self.batch_size, self.f_vocab_size]))

		return logits, predictions
		#self.predictions = tf.nn.softmax(logits, name='predictions')
		
		#These three lines will be moved to the loss_op method
		#logits1 = tf.unpack(tf.reshape(logits, [self.max_seq_length, self.batch_size, self.f_vocab_size]))


		#f_targets = tf.split(1, self.max_seq_length, f_targets)
		#f_targets = [tf.squeeze(sequence_, [1]) for sequence_ in f_targets]
		#loss = seq2seq.sequence_loss_by_example(logits1, f_targets, [tf.ones([batch_size]) for i in range(max_seq_length)])
		#self.cost = tf.reduce_sum(loss) / batch_size

	def inference_e_f(self, e_sequences, e_seq_masks, f_sequences, f_seq_masks, f_targets, feed_previous) :

		""" Following SMT literature, the prefixes e_ and f_ are just placeholders, you can replace them by any language pair
		defines the following model:
		All the comment assume that language 1 is en and language 2 is fr 
		1) compute hidden representation (h_e) for en treating e_sequences as bag-of-words [h_e = g(W_emb_e * n_hot_e_rep + b)]
		(n_hot_e_rep is a |V| size vector where all the words in e_sequence are hot)
		2) compute hidden representation (h_f) for fr treating f_sequences as bag-of-words 
		3) reconstruct en bag-of-words from h_e and h_f 
		4) reconstruct fr bag-of-words from h_e and h_f
		5) reconstruct fr "sequence" from h_e and h_f

		:params:
		  e_sequences : tensor, int32 - [batch_size, max_seq_length]
		    indices of the words in the English sentences
		    if the English sentence is "I am at home" and max_seq_length is 8 then the sentence effectively becomes
		    "<GO> I am at home <EOS> <PAD> <PAD>". e_sequences will then be an array of the integer ids of these words 
		    "[id(<GO>),id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]"
		    The same process is repeated for all sentences in the batch	to get a tensor of size [batch_size, max_seq_length]
		  e_seq_masks : tensor, int32 - [batch_size, max_seq_length]
		    each entry in this tensor indicates whether that word should be ignored or not. For example, for computing h_en
		    we simply sum up the embeddings of all words in the sentence except <PAD> and <GO>. Thus for the above example of  
		    "<GO> I am at home <EOS> <PAD> <PAD>" the seq_masks will be [0,1,1,1,1,1,0,0].
		    The same process is repeated for all sentences in the batch	to get a tensor of size [batch_size, max_seq_length]
		  f_sequences : tensor, int32 - [batch_size, max_seq_length]
		    same logic as e_sequences
		  f_seq_masks : tensor, int32 - [batch_size, max_seq_length]
		  	same logic as e_seq_masks
		  f_targets : tensor, int32 - [batch_size, max_seq_length]
		  	these are the indices of the target words to be decoded. if the input sentence is 
		  	"[id(<GO>),id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]" then the targets are simply shifted by 1
		  	"[id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]". Essentially the model sees <GO> and produces the 
		  	target I, then sees I and produces am and so on. See the figure in "Sequence to Sequence basics" section here:
		  	https://www.tensorflow.org/versions/r0.7/tutorials/seq2seq/index.html	
		  feed_previous: Boolean or scalar Boolean Tensor; 
		  Should be set to True during training and False during testing
		  see feed_previous seq2seq.embedding_rnn_decoder for more details
		:return:
		  h_e : tensor, int32 - [batch_size, embedding_size]
		  the hidden representation computed for e_vocab_size

		  h_f : tensor, int32 - [batch_size, embedding_size]
		  the hidden representation computed for f_vocab_size
		  
		  e2f: tensor, int32 - [batch_size, f_vocab_size]
		  f' reconstructed from the hidden representation of h_e.	
		  
		  f2f: tensor, int32 - [batch_size, f_vocab_size]
		  f' reconstructed from the hidden representation of h_f.	
		  
		  e2e: tensor, int32 - [batch_size, e_vocab_size]
		  e' reconstructed from the hidden representation of h_e.	
		  
		  f2e: tensor, int32 - [batch_size, f_vocab_size]
		  e' reconstructed from the hidden representation of h_f.	

		  e2f_seq: List of 2D Tensors of shape [batch_size x f_vocab_size]. the length of the list is equal to max_seq_length
		  The i-th row of the j-th tensor in the list contains the logit values calculated for the j-th word in the i-th 
		  sentence in the batch [go figure!!]. These logits were calculated by passing thorugh an rnn decoder which takes h_e as h_zero  
		  
		  f2f_seq: List of 2D Tensors of shape [batch_size x f_vocab_size]. the length of the list is equal to max_seq_length
		  The i-th row of the j-th tensor in the list contains the logit values calculated for the j-th word in the i-th 
		  sentence in the batch [go figure!!]. These logits were calculated by passing thorugh an rnn decoder which takes h_f as h_zero  
 		"""

		"""
		# ========================================================================================
		# COMPUTE h_e [batch_size, embedding_size]: the hidden representation of the en sentences
		# ========================================================================================
		"""
		h_e = self.compute_hidden_representation(e_sequences, e_seq_masks, self.W_emb_e)

		"""
		# ========================================================================================
		# COMPUTE h_f [batch_size, embedding_size]: the hidden representation of the fr sentences
		# ========================================================================================
		"""
		h_f = self.compute_hidden_representation(f_sequences, f_seq_masks, self.W_emb_f)

		"""
		# =========================================================================================================================
		# RECONSTRUCT the original fr bag-of-words (bow) from h_e : e2f [batch_size, f_vocab_size] (e2f means use h_e to decode f)
		# =========================================================================================================================
		"""
		e2f = self.reconstruct_bow_from_h(h_e, tf.transpose(self.W_emb_f), self.b_dec_f)	#encoder & decoder weights are tied, hence taking transpose
		"""
		# =========================================================================================================================
		# RECONSTRUCT the original fr bag-of-words (bow) from h_f : f2f [batch_size, f_vocab_size] (f2f means use h_f to decode f)
		# =========================================================================================================================
		"""
		f2f = self.reconstruct_bow_from_h(h_f, tf.transpose(self.W_emb_f), self.b_dec_f)	
		"""
		# ==================================================================================================================
		# RECONSTRUCT the original en bag-of-words (bow) from h_e : e2e [batch_size, e_vocab_size] 
		# ==================================================================================================================
		"""
		e2e = self.reconstruct_bow_from_h(h_e, tf.transpose(self.W_emb_e), self.b_dec_e)	
		"""
		# ==================================================================================================================
		# RECONSTRUCT the original en bag-of-words (bow) from h_f : f2e [batch_size, e_vocab_size] 
		# ==================================================================================================================
		"""
		f2e = self.reconstruct_bow_from_h(h_f, tf.transpose(self.W_emb_e), self.b_dec_e)

		"""
		# ==================================================================================================================
		# CONVERT f_sequences to a format accepted by seq2seq.embedding_rnn_decoder
		# ==================================================================================================================
		"""
		#the next two lines will convert f_sequences (batch_size * max_seq_length) to a list of
		#tensors where each tensor is of size (batch_size). The length of the list is same as max_seq_length
		#this is the format accepted by seq2seq.embedding_rnn_decoder
		f_sequences_to_be_decoded = tf.split(1, self.max_seq_length, f_sequences)
		f_sequences_to_be_decoded = [tf.squeeze(sequence_, [1]) for sequence_ in f_sequences_to_be_decoded]

		with tf.variable_scope("root") as scope:
			"""
			# =========================================================================================================================
			# DECODE the original fr sequence (seq) by feeding h_f as h_0 (enc_state) to the decoder : f2f_seq
			# this is a max_seq_length sized list of tensors where each tensor is of size (batch_size, f_vocab_size). It is basically
			# the logits vector from which the probabilities can be inferred by doing a softmax
			# =========================================================================================================================
			"""
			f2f_logits, f2f_pred_seq = self.decode_f_seq_from_h(f_sequences_to_be_decoded, h_f, 
												self.f_cell, self.f_vocab_size, self.W_f_rnn_proj, self.b_f_rnn_proj, feed_previous)
			"""
			# ==========================================================================================================================
			# DECODE the original fr sequence (seq) by feeding h_e as h_0 (enc_state) to the decoder : e2f_seq
			# this is a max_seq_length sized list of tensors where each tensor is of size (batch_size, f_vocab_size). It is basically
			# the logits vector from which the probabilities can be inferred by doing a softmax
			# ==========================================================================================================================
			"""
			scope.reuse_variables()
			e2f_logits, e2f_pred_seq = self.decode_f_seq_from_h(f_sequences_to_be_decoded, h_e, 
												self.f_cell, self.f_vocab_size, self.W_f_rnn_proj, self.b_f_rnn_proj, feed_previous)

		return h_e, h_f, e2f, f2f, e2e, f2e, e2f_logits, e2f_pred_seq, f2f_logits, f2f_pred_seq

	def loss_e_f(self, e_bow, f_bow, f_targets, h_e, h_f, e2f, f2f, e2e, f2e, e2f_seq, f2f_seq) :
		""" Computes the loss for reconstructing e', f' and f_seq' for an e-f pair
		:params:
		  e_bow : tensor, int32 - [batch_size, e_vocab_size]
		    a n-hot encoding of the en sentences. Each row corresponds to one sentence where the dimensions corresponding to the
		    words present in the sentence are hot (set to 1)

		  f_bow : tensor, int32 - [batch_size, f_vocab_size]
		    a n-hot encoding of the fr sentences. Each row corresponds to one sentence where the dimensions corresponding to the
		    words present in the sentence are hot (set to 1)

		  f_targets : tensor, int32 - [batch_size, max_seq_length]
		  	these are the indices of the target words to be decoded. if the input sentence is 
		  	"[id(<GO>),id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]" then the targets are simply shifted by 1
		  	"[id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]". Essentially the model sees <GO> and produces the 
		  	target I, then sees I and produces am and so on. See the figure in "Sequence to Sequence basics" section here:
		  	https://www.tensorflow.org/versions/r0.7/tutorials/seq2seq/index.html	

		  h_e : tensor, int32 - [batch_size, embedding_size]
		  the hidden representation computed for e_sequences [see self.inference_e_f]

		  h_f : tensor, int32 - [batch_size, embedding_size]
		  the hidden representation computed for f_sequences [see self.inference_e_f]
		  
		  e2f: tensor, int32 - [batch_size, f_vocab_size]
		  f' reconstructed from the hidden representation of h_e [see self.inference_e_f]
		  
		  f2f: tensor, int32 - [batch_size, f_vocab_size]
		  f' reconstructed from the hidden representation of h_f [see self.inference_e_f]
		  
		  e2e: tensor, int32 - [batch_size, e_vocab_size]
		  e' reconstructed from the hidden representation of h_e [see self.inference_e_f]
		  
		  f2e: tensor, int32 - [batch_size, f_vocab_size]
		  e' reconstructed from the hidden representation of h_f [see self.inference_e_f]

		  e2f_seq: List of 2D Tensors of shape [batch_size x f_vocab_size]. the length of the list is equal to max_seq_length
		  The i-th row of the j-th tensor in the list contains the logit values calculated for the j-th word in the i-th 
		  sentence in the batch [go figure!!]. These logits were calculated by passing thorugh an rnn decoder which takes h_e as h_zero  
		  
		  f2f_seq: List of 2D Tensors of shape [batch_size x f_vocab_size]. the length of the list is equal to max_seq_length
		  The i-th row of the j-th tensor in the list contains the logit values calculated for the j-th word in the i-th 
		  sentence in the batch [go figure!!]. These logits were calculated by passing thorugh an rnn decoder which takes h_f as h_zero  

		:return:
		  l_e2f : float32
		  the squared error loss between e2f and f_bow

		  l_f2f : float32
		  the squared error loss between f2f and f_bow

		  l_e2e : float32
		  the squared error loss between e2e and e_bow 	  

		  l_f2e: float32
		  the squared error loss between f2e and e_bow

		  l_e2f_seq: float32
		  the cross entropy loss between e2f_seq and f_targets

		  l_f2f_seq: float32
		  the cross entropy loss between f2f_seq and f_targets
 		"""

 		l_e2f = self.compute_squared_error_loss(e2f, f_bow)	

 		l_f2f = self.compute_squared_error_loss(f2f, f_bow)	

 		l_e2e = self.compute_squared_error_loss(e2e, e_bow)	

 		l_f2e = self.compute_squared_error_loss(f2e, e_bow)

		"""
		# ==================================================================================================================
		# CONVERT f_targets to a format accepted by seq2seq.sequence_loss_by_example
		# ==================================================================================================================
		"""
 		f_targets = tf.split(1, self.max_seq_length, f_targets)
		f_targets = [tf.squeeze(sequence_, [1]) for sequence_ in f_targets]
		
		l_e2f_seq = seq2seq.sequence_loss_by_example(e2f_seq, f_targets, [tf.ones([self.batch_size]) for i in range(self.max_seq_length)])
		l_e2f_seq = tf.reduce_sum(l_e2f_seq) / self.batch_size

		l_f2f_seq = seq2seq.sequence_loss_by_example(f2f_seq, f_targets, [tf.ones([self.batch_size]) for i in range(self.max_seq_length)])
		l_f2f_seq = tf.reduce_sum(l_f2f_seq) / self.batch_size

		loss = l_e2f + l_f2f + l_e2e + l_f2e + l_e2f_seq + l_f2f_seq
		
		return l_e2f, l_f2f, l_e2e, l_f2e, l_e2f_seq, l_f2f_seq, loss

	def training_e_f(self,loss, optimizer = 'adam', learning_rate = 1e-3):
		""" sets up the training ops

		:params:
		  loss: loss tensor, from loss()
		  optimizer: str
			gradient_descent, adam
		  learning_rate : float
        the learning rate  
		:returns:
		  train_op: the op for training
		"""
		# create a variable to track the global step.
		global_step = tf.Variable(0, name='global_step', trainable=False)

		# create the gradient descent optimizer with the given learning rate.
		if optimizer == 'gradient_descent':
			optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		elif optimizer == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate)
		else:
			raise Exception("optimizer type not supported: {}".format(optimizer))
  
		# use the optimizer to apply the gradients that minimize the loss
		# (and also increment the global step counter) as a single training step.
		#train_op = optimizer.minimize(loss, global_step=global_step)

		grads_and_vars = optimizer.compute_gradients(loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		return train_op


