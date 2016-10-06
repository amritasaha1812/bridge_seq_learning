import tensorflow as tf
import numpy as np
#from tensorflow.models.rnn import rnn,rnn_cell,seq2seq
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import array_ops
import rnn_cell1, seq2seq1
import scipy as sp
from itertools import chain
import os
import time
import datetime

class BridgeCaptionsModel():
  """ test model for bridge caption generation
  """

  def __init__(self, max_seq_length, embedding_size, batch_size, rnn_size, e_vocab_size,
          f_vocab_size, image_size, lmbda, random_seed, activation=sigmoid,
          e_bias_init_vector=None, f_bias_init_vector=None):
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
      self.lmbda = lmbda

      """
      # ==================================================
      # DECLARE ALL THE PARAMETERS OF THE MODEL
      # ==================================================
      """
      #after training this will contain the word embeddings for the words in f
      max_val = np.sqrt(6. / (self.image_size + self.embedding_size))
      self.W_emb_im = tf.Variable(tf.random_uniform([self.image_size, self.embedding_size], -1.* max_val, max_val), name="W_emb_im")
      
      max_val = np.sqrt(6. / (self.e_vocab_size + self.embedding_size))
      self.W_emb_e = tf.Variable(tf.random_uniform([self.e_vocab_size, self.embedding_size], -1.* max_val, max_val), name="W_emb_e")
      
      #the encoder bias is common for image and en
      self.b_enc_common = tf.Variable(tf.constant(0., shape=[self.embedding_size]), name='b_enc_common')

      with tf.device("/cpu:0"):
         self.W_f_rnn_emb = tf.Variable(tf.random_uniform([self.f_vocab_size, self.rnn_size],
             -0.1, 0.1), name='W_f_rnn_emb')
     
      self.b_f_rnn_emb = tf.Variable(tf.constant(0., shape=[self.rnn_size]), name='b_f_rnn_emb')

      self.f_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)

      self.W_f_rnn_out = tf.Variable(tf.random_uniform([self.rnn_size, self.f_vocab_size], -0.1, 0.1), name='W_f_rnn_out')
      
      if f_bias_init_vector is not None:
         self.b_f_rnn_out = tf.Variable(f_bias_init_vector.astype(np.float32), name='b_f_rnn_out')
      else:
         self.b_f_rnn_out = tf.Variable(tf.constant(0., shape=[self.f_vocab_size]), name='b_f_rnn_out')

  def standardize(self, x, mean, stddev, dim0, dim1) :
    """
    x : tensor, float32 [None, None]
    mean : tensor, float32 [None] : dimension of mean is same as the second dimension of x [i.e., dim1]
    stddev : tensor, float32 [None] : dimension of stddev is same as the second dimension of x [i.e., dim1]
    :return: tensor, float32 [None, None]
    (x - mean) / stddev
    """
    #x_mean = tf.reshape(tf.tile(mean, [dim0]), [dim0, dim1])
    #x_stdev = tf.reshape(tf.tile(stddev, [dim0]), [dim0, dim1])
    x_centered = tf.sub(x, mean)
    x_std = tf.div(x_centered, stddev)

    return x_std

  def compute_correlation(self, x, y) :
    """
    x : tensor, float32 [None, None]
    y : tensor, float32 [None, None] (same shape as x)
    :return: tensor, float32 
    correlation between x and y 
    """
    x_mean = tf.reshape(tf.tile(tf.reduce_mean(x, 0), [self.batch_size]), [self.batch_size, self.embedding_size])
    y_mean = tf.reshape(tf.tile(tf.reduce_mean(y, 0), [self.batch_size]), [self.batch_size, self.embedding_size])

    x_centered = tf.sub(x, x_mean)
    y_centered = tf.sub(y, y_mean)

    corr_nr = tf.reduce_sum(tf.mul(x_centered, y_centered), 0)
    corr_dr1 = tf.sqrt(tf.reduce_sum(tf.mul(x_centered, x_centered), 0))
    corr_dr2 = tf.sqrt(tf.reduce_sum(tf.mul(y_centered, y_centered), 0))

    corr_dr = tf.mul(corr_dr1, corr_dr2)  

    corr = tf.div(corr_nr, corr_dr)

    #return x_centered, y_centered, corr_nr, corr_dr1, corr_dr2, corr_dr, tf.reduce_sum(corr)
    return corr, tf.reduce_sum(corr)


  def compute_hidden_representation1(self, sequences, seq_masks, W) :
    """ 
    :params:
      sequences : tensor, int32 - [batch_size, max_seq_length]
        indices of the words in the English sentences
        if the English sentence is "I am at home" and max_seq_length is 8 then the sentence effectively becomes
        "<GO> I am at home <EOS> <PAD> <PAD>". e_sequences will then be an array of the integer ids of these words 
        "[id(<GO>),id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]"
        The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
      seq_masks : tensor, int32 - [batch_size, max_seq_length]
        each entry in this tensor indicates whether that word should be ignored or not. For example, for computing h_en
        we simply sum up the embeddings of all words in the sentence except <PAD> and <GO>. Thus for the above example of  
        "<GO> I am at home <EOS> <PAD> <PAD>" the seq_masks will be [0,1,1,1,1,1,0,0].
      The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
      W: tensor, float32 - [vocab_size, embedding_size] or [image_size, embedding_size]
        the projection matrix (W_emb_en or W_emb_fr or W_emb_im)
    :return: tensor, int32 - [batch_size, embedding_size]
      h : the computed hidden representation of the input   
    """
    # the next line will return a list of tensors where each tensor corresponds
    # to one sentence and is of dimension (max_seq_length * embedding_size)
    # the length of the list is same as self.batch_size
    sequences = tf.split(0, 1, tf.nn.embedding_lookup(W, sequences)) # 1= batch_size
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
    #     [2,2] <-- (embeddding of 2nd word in the sentence)
    #     [3,3] <-- (embeddding of <PAD>)
    #     [3,3] <-- (embeddding of <PAD>)
    # = [3,2] (then add the bias to this)
    # h is a tensor of size (batch_size, embedding_size)
    #there should be a way to optimize the next two lines
    h = [tf.nn.xw_plus_b(tf.reshape(seq_masks[i], (1,self.max_seq_length)), sequences[i], self.b_enc_common) for i in range(len(sequences))] 
    h = tf.squeeze(tf.pack(h), [1]) #the reason for squeezing is same as mentioned above

    return h

  def compute_hidden_representation(self, sequences, seq_masks, W) :
    """ 
    :params:
      sequences : tensor, int32 - [batch_size, max_seq_length]
        indices of the words in the English sentences
        if the English sentence is "I am at home" and max_seq_length is 8 then the sentence effectively becomes
        "<GO> I am at home <EOS> <PAD> <PAD>". e_sequences will then be an array of the integer ids of these words 
        "[id(<GO>),id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]"
        The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
      seq_masks : tensor, int32 - [batch_size, max_seq_length]
        each entry in this tensor indicates whether that word should be ignored or not. For example, for computing h_en
        we simply sum up the embeddings of all words in the sentence except <PAD> and <GO>. Thus for the above example of  
        "<GO> I am at home <EOS> <PAD> <PAD>" the seq_masks will be [0,1,1,1,1,1,0,0].
      The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
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
    #     [2,2] <-- (embeddding of 2nd word in the sentence)
    #     [3,3] <-- (embeddding of <PAD>)
    #     [3,3] <-- (embeddding of <PAD>)
    # = [3,2] (then add the bias to this)
    # h is a tensor of size (batch_size, embedding_size)
    #there should be a way to optimize the next two lines
    h = [tf.nn.xw_plus_b(tf.reshape(seq_masks[i], (1,self.max_seq_length)), sequences[i], self.b_enc_common) for i in range(len(sequences))] 
    h = tf.squeeze(tf.pack(h), [1]) #the reason for squeezing is same as mentioned above

    return h

  def compute_hidden_representation2(self, sequences, seq_masks, W) :
    """ 
    :params:
      sequences : tensor, int32 - [batch_size, max_seq_length]
        indices of the words in the English sentences
        if the English sentence is "I am at home" and max_seq_length is 8 then the sentence effectively becomes
        "<GO> I am at home <EOS> <PAD> <PAD>". e_sequences will then be an array of the integer ids of these words 
        "[id(<GO>),id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]"
        The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
      seq_masks : tensor, int32 - [batch_size, max_seq_length]
        each entry in this tensor indicates whether that word should be ignored or not. For example, for computing h_en
        we simply sum up the embeddings of all words in the sentence except <PAD> and <GO>. Thus for the above example of  
        "<GO> I am at home <EOS> <PAD> <PAD>" the seq_masks will be [0,1,1,1,1,1,0,0].
      The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
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
    #     [2,2] <-- (embeddding of 2nd word in the sentence)
    #     [3,3] <-- (embeddding of <PAD>)
    #     [3,3] <-- (embeddding of <PAD>)
    # = [3,2] (then add the bias to this)
    # h is a tensor of size (batch_size, embedding_size)
    #there should be a way to optimize the next two lines
    h = [tf.nn.xw_plus_b(tf.reshape(seq_masks[i], (1,self.max_seq_length)), sequences[i], self.b_enc_common) for i in range(len(sequences))] 
    h = tf.squeeze(tf.pack(h), [1]) #the reason for squeezing is same as mentioned above

    return h

  def get_seq_loss(self, target_sequences, target_masks, target_vocab_size, enc_state, cell, 
    W_rnn_emb, b_rnn_emb, W_rnn_out, b_rnn_out, scope) :
    """ 
    :params:
    """ 
    state = tf.zeros([self.batch_size, cell.state_size])

    loss = 0.0
    with tf.variable_scope(scope):
      for i in range(self.max_seq_length): # maxlen + 1
        if i == 0:
          current_emb = enc_state
        else:
          with tf.device("/cpu:0"):
            current_emb = tf.nn.embedding_lookup(W_rnn_emb, target_sequences[:,i-1]) + b_rnn_emb

        if i > 0 : tf.get_variable_scope().reuse_variables()

        output, state = cell(current_emb, state) # (batch_size, dim_hidden)

        if i > 0: 
          labels = tf.expand_dims(target_sequences[:, i], 1) # (batch_size)
          indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
          concated = tf.concat(1, [indices, labels])
          onehot_labels = tf.sparse_to_dense(
                            concated, tf.pack([self.batch_size, target_vocab_size]), 1.0, 0.0) # (batch_size, n_words)

          logit_words = tf.matmul(output, W_rnn_out) + b_rnn_out # (batch_size, n_words)
          cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
          cross_entropy = cross_entropy * target_masks[:,i]#tf.expand_dims(mask, 1)

          current_loss = tf.reduce_sum(cross_entropy)
          loss = loss + current_loss

      loss = loss / tf.reduce_sum(target_masks[:,1:])
      
      return loss
  
  def get_standard_i(self, image1, img_mean, img_stddev, selection_matrix) :
    h_i = sigmoid(tf.nn.xw_plus_b(image1, self.W_emb_im, self.b_enc_common))
    h_i = self.standardize(h_i, img_mean, img_stddev, 1, self.embedding_size)
    h_i = tf.matmul(h_i, selection_matrix)
    return tf.reshape(h_i, (1, self.rnn_size))

  def inference_i2f(self, image1, img_mean, img_stddev, selection_matrix) :
    h_i = sigmoid(tf.nn.xw_plus_b(image1, self.W_emb_im, self.b_enc_common))
    h_i = self.standardize(h_i, img_mean, img_stddev, 1, self.embedding_size)
    h_i = tf.matmul(h_i, selection_matrix)
    state = tf.zeros([1, self.f_lstm.state_size])
        
    generated_words = []

    with tf.variable_scope("fr_lstm"):
      tf.get_variable_scope().reuse_variables()
      output, state = self.f_lstm(h_i, state)
      last_word = tf.nn.embedding_lookup(self.W_f_rnn_emb, [1]) + self.b_f_rnn_emb

      for i in range(self.max_seq_length):

        output, state = self.f_lstm(last_word, state)

        logit_words = tf.matmul(output, self.W_f_rnn_out) + self.b_f_rnn_out
        max_prob_word = tf.argmax(logit_words, 1)

        with tf.device("/cpu:0"):
          last_word = tf.nn.embedding_lookup(self.W_f_rnn_emb, max_prob_word)

        last_word += self.b_f_rnn_emb

        generated_words.append(max_prob_word)

      #return h_i, generated_words
      return generated_words

  def inference_e2f(self, e_sequences, e_seq_masks, e_mean, e_stddev, selection_matrix):
    h_e = sigmoid(self.compute_hidden_representation1(e_sequences, e_seq_masks, self.W_emb_e))
    h_e = self.standardize(h_e, e_mean, e_stddev, 1, self.embedding_size)
    h_e = tf.matmul(h_e, selection_matrix)
    state = tf.zeros([1, self.f_lstm.state_size])
        
    generated_words = []

    with tf.variable_scope("fr_lstm"):
      tf.get_variable_scope().reuse_variables()
      output, state = self.f_lstm(h_e, state)
      last_word = tf.nn.embedding_lookup(self.W_f_rnn_emb, [1]) + self.b_f_rnn_emb

      for i in range(self.max_seq_length):

        output, state = self.f_lstm(last_word, state)

        logit_words = tf.matmul(output, self.W_f_rnn_out) + self.b_f_rnn_out
        max_prob_word = tf.argmax(logit_words, 1)

        with tf.device("/cpu:0"):
          last_word = tf.nn.embedding_lookup(self.W_f_rnn_emb, max_prob_word)

        last_word += self.b_f_rnn_emb

        generated_words.append(max_prob_word)

      return generated_words
  
  def loss_e2f(self, e_sequences, e_seq_masks, f_sequences, f_seq_masks, e_mean, e_stddev, selection_matrix) :

    """ Following SMT literature, the prefixes e_ and f_ are just placeholders, you can replace them by any language pair
    defines the following model:
    All the comment assume that language 1 is en and language 2 is fr 
    1) compute hidden representation (h_e) for en treating e_sequences as bag-of-words [h_e = g(W_emb_e * n_hot_e_rep + b)]
    (n_hot_e_rep is a |V| size vector where all the words in e_sequence are hot)
    2) construct fr "sequence" from h_e
    3) calculate cross-entropy loss between true f_sequences and predicted f_sequences

    :params:
      e_sequences : tensor, int32 - [batch_size, max_seq_length]
        indices of the words in the English sentences
        if the English sentence is "I am at home" and max_seq_length is 8 then the sentence effectively becomes
        "<GO> I am at home <EOS> <PAD> <PAD>". e_sequences will then be an array of the integer ids of these words 
        "[id(<GO>),id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]"
        The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
      e_seq_masks : tensor, int32 - [batch_size, max_seq_length]
        each entry in this tensor indicates whether that word should be ignored or not. For example, for computing h_en
        we simply sum up the embeddings of all words in the sentence except <PAD> and <GO>. Thus for the above example of  
        "<GO> I am at home <EOS> <PAD> <PAD>" the seq_masks will be [0,1,1,1,1,1,0,0].
        The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
      f_sequences : tensor, int32 - [batch_size, max_seq_length]
        same logic as e_sequences
      f_seq_masks : tensor, int32 - [batch_size, max_seq_length]
        same logic as e_seq_masks
    :return:
      h_e : tensor, int32 - [batch_size, embedding_size]
      the hidden representation computed for e_sequences

      l_e2f : tensor, int32 
      the average cross entropy loss between predicted f_sequences and true f_sequences (average over seq length and 
        further average over batch_size)
    """

    """
    # ========================================================================================
    # COMPUTE h_e [batch_size, embedding_size]: the hidden representation of the en sentences
    # ========================================================================================
    """
    h_e = sigmoid(self.compute_hidden_representation(e_sequences, e_seq_masks, self.W_emb_e))
    h_e = self.standardize(h_e, e_mean, e_stddev, self.batch_size, self.embedding_size)
    h_e = tf.matmul(h_e, selection_matrix)

    """
    # ==================================================================================================================
    # CALCULATE cross-entropy loss of constructing f_sequences from h_e
    # ==================================================================================================================
    """
    l_e2f = self.get_seq_loss(f_sequences, f_seq_masks, self.f_vocab_size, h_e, self.f_lstm, 
      self.W_f_rnn_emb, self.b_f_rnn_emb, self.W_f_rnn_out, self.b_f_rnn_out, "fr_lstm")
    
    return h_e, l_e2f


  def loss_e_i(self, e_sequences, e_seq_masks, images, lmbda, weights) :

    """ 
    1) compute hidden representation (h_e) for en treating e_sequences as bag-of-words [h_e = g(W_emb_e * n_hot_e_rep + b)]
    (n_hot_e_rep is a |V| size vector where all the words in e_sequence are hot)
    2) compute hidden representation (h_i) for image by passing it through a feeedforward layer
    3) calculate correlation loss between h_e and h_i

    :params:
      e_sequences : tensor, int32 - [batch_size, max_seq_length]
        indices of the words in the English sentences
        if the English sentence is "I am at home" and max_seq_length is 8 then the sentence effectively becomes
        "<GO> I am at home <EOS> <PAD> <PAD>". e_sequences will then be an array of the integer ids of these words 
        "[id(<GO>),id(I,id(am),id(at),id(home),id(<EOS>),id(<PAD>),id(<PAD>)]"
        The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
      e_seq_masks : tensor, int32 - [batch_size, max_seq_length]
        each entry in this tensor indicates whether that word should be ignored or not. For example, for computing h_en
        we simply sum up the embeddings of all words in the sentence except <PAD> and <GO>. Thus for the above example of  
        "<GO> I am at home <EOS> <PAD> <PAD>" the seq_masks will be [0,1,1,1,1,1,0,0].
        The same process is repeated for all sentences in the batch to get a tensor of size [batch_size, max_seq_length]
      images : tensor, int32 - [batch_size, image_size]
        a batch of images
      lambda : tensor, float32
        weight for the correlation loss  
    :return:
      l_ei_corr : tensor, float32
      the sum correlation between h_e and h_i

      loss : tensor, float32
      the total loss which is simply lambda * l_ei_corr

      h_e : tensor, int32 - [batch_size, embedding_size]
      the hidden representation computed for e_sequences

      h_i : tensor, int32 - [batch_size, embedding_size]
      the hidden representation computed for images
    """

    """
    # ========================================================================================
    # COMPUTE h_e [batch_size, embedding_size]: the hidden representation of the en sentences
    # ========================================================================================
    """
    h_e = sigmoid(self.compute_hidden_representation(e_sequences, e_seq_masks, self.W_emb_e))

    """
    # ========================================================================================
    # COMPUTE h_i [batch_size, embedding_size]: the hidden representation of the images
    # ========================================================================================
    """
    h_i = sigmoid(tf.nn.xw_plus_b(images, self.W_emb_im, self.b_enc_common))

    """
    # ==================================================================================================================
    # CALCULATE correlation between h_e and h_i
    # ==================================================================================================================
    """
    corr_vec, l_ei_corr = self.compute_correlation(h_e, h_i)

    loss = tf.reduce_sum(tf.sub(weights, corr_vec))
    
    #weighted_corr = tf.mul(corr_vec, weights)

    #l_ei_corr1 = tf.reduce_sum(weighted_corr)
    
    #loss = 0 - lmbda * l_ei_corr1
    return l_ei_corr, loss, h_e, h_i, corr_vec

  def training_e_f(self, loss, optimizer = 'adam', learning_rate = 1e-3):
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
    capped_grads_and_vars = [(tf.clip_by_value(gv[0], -0.1, 0.1), gv[1]) for gv in grads_and_vars if gv[0] != None]
    #capped_grads_and_vars = [(gv[0], gv[1]) for gv in grads_and_vars if gv[1].name != 'W_emb_e:0'  and gv[1].name != 'b_enc_common:0']
    train_e_f_op = optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)

    return train_e_f_op

  def training_e_i(self, loss, optimizer = 'adam', learning_rate = 1e-3):
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
    #train_e_i_op = optimizer.minimize(loss)  
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_grads_and_vars = [(tf.clip_by_value(gv[0], -0.1, 0.1), gv[1]) for gv in grads_and_vars if gv[0] != None]
    train_e_i_op = optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)

    return train_e_i_op

