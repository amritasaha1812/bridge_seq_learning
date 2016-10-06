import tensorflow as tf
#from tensorflow.models.rnn import seq2seq
import numpy as np

def decode(probabilities) :
	return tf.transpose(tf.argmax(probabilities, 2))

def train(seq_inputs, W) :
	x = tf.nn.embedding_lookup(W, seq_inputs)
	inputs1 = tf.split(1, 3, tf.nn.embedding_lookup(W, seq_inputs))
	inputs = [tf.squeeze(input_, [1]) for input_ in inputs1]
	inputs = [tf.constant(0.5, shape=[2, 2])] * 2
	inputs = [tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
	#inputs = [tf.reduce_sum(input_, [0]) for input_ in inputs]
	return inputs

def calculate_correlation(x, y, batch_size, embedding_size) :

	x_mean = tf.reshape(tf.tile(tf.reduce_mean(x, 0), [batch_size]), [batch_size, embedding_size])
	y_mean = tf.reshape(tf.tile(tf.reduce_mean(y, 0), [batch_size]), [batch_size, embedding_size])

	#x_mean = tf.reshape(x_mean, [batch_size, embedding_size])
	#y_mean = tf.reshape(y_mean, [batch_size, embedding_size])

	x_centered = tf.sub(x, x_mean)
	y_centered = tf.sub(y, y_mean)

	corr_nr = tf.reduce_sum(tf.mul(x_centered, y_centered), 0)
	corr_dr1 = tf.sqrt(tf.reduce_sum(tf.mul(x_centered, x_centered), 0))
	corr_dr2 = tf.sqrt(tf.reduce_sum(tf.mul(y_centered, y_centered), 0))

	corr_dr = tf.mul(corr_dr1, corr_dr2)

	corr = tf.div(corr_nr, corr_dr)
	cov_x = compute_cov_matrix(x_centered, embedding_size)
	cov_y = compute_cov_matrix(y_centered, embedding_size)
    	cov = ( cov_x + cov_y )/2.0
	corr_sum = tf.reduce_sum(corr) - cov	
	return x_centered, y_centered, corr_nr, corr_dr1, corr_dr2, corr_dr, corr_sum

def compute_cov_matrix(x_centered, num_dims):
   sum_cov = 0.0
   count= 0.0
   var = tf.reduce_sum(tf.mul(x_centered, x_centered), 0)
   for i in xrange(num_dims):
         for j in range(i+1, num_dims):
            r_num = tf.reduce_sum(tf.mul(x_centered[:,i], x_centered[:,j]), 0)
            r_den = tf.sqrt(var[i]*var[j])
            sum_cov = sum_cov + tf.abs(r_num/r_den)
            count = count + 1
   return sum_cov/count

def test(dec_inp, enc_state, cell) :
	#dec_inp = [tf.split(0, 3, dec_inp)]
	dec_inp = tf.split(1, 3, dec_inp)
	inputs = [tf.squeeze(input_, [1]) for input_ in dec_inp]
	dec, mem = tf.nn.seq2seq.embedding_rnn_decoder(inputs, enc_state, cell, 4, 3)
	return mem

if __name__ == "__main__":
	#dec_inp  = []
	#for i in range(3) :
	#	dec_inp.append(tf.placeholder(tf.int32, shape=(3), name="seq_inputs"))
	dec_inp = tf.placeholder(tf.int32, shape=(4,3), name="seq_inputs")
	x = tf.placeholder(tf.float32, shape=(2,5), name="inpx")
	y = tf.placeholder(tf.float32, shape=(2,5), name="inpy")
	probs = tf.placeholder(tf.float32, shape=(4,3,5), name="inpy")
	decoder_inputs = np.asarray([[0,1,2],[2,1,0],[2,1,1],[1,0,1]])
	inp_x = np.asarray([[0.,1.,2.],[2.,1.,0.],[2.,1.,1.],[1.,0.,1.]])
	inp_y = np.asarray([[0.,2.,4.],[4.,2.,0.],[4.,2.,2.],[2.,0.,2.]])

	#(4,3,5) 
	inp_probs = [[[0.01, 0.95, 0.02, 0.01, 0.05], [0.00, 0.00, 0.015, 0.98, 0.005], [0.85, 0.13, 0.005, 0.01, 0.005]], 
		[[0.00, 0.00, 0.015, 0.98, 0.005], [0.01, 0.95, 0.02, 0.01, 0.05], [0.85, 0.13, 0.005, 0.01, 0.005]],
		[[0.01, 0.95, 0.02, 0.01, 0.05], [0.85, 0.13, 0.005, 0.01, 0.005], [0.00, 0.00, 0.015, 0.98, 0.005]],
		[[0.01, 0.95, 0.98, 0.01, 0.05], [1.00, 0.00, 0.015, 0.98, 0.005], [0.85, 0.13, 0.005, 0.01, 1.005]]]

	inp_x = np.asarray([[-0.00744113,-0.01054093,0.01781106,-0.03446457,0.03112096], [0.00744113,0.01054093,-0.01781106,0.03446454,-0.03112096]])
	inp_y = np.asarray([[-0.01096278,0.03095388,0.04766631,-0.03565353,-0.02007136], [0.01096275,-0.03095382,-0.04766631,0.03565353,0.02007136]])
 
	batch_decoder_inputs = []
	for length_idx in xrange(3): # 3 is max sequence length
		batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(4)], dtype=np.int32)) # 4 is batch size
	#print decoder_inputs
	#print batch_decoder_inputs

	#print inp_x 
	#print inp_y
	print inp_probs

	session_conf = tf.ConfigProto(allow_soft_placement=True)

	x_op, y_op, corr_nr_op, corr_dr1_op, corr_dr2_op, corr_dr_op, corr_op = calculate_correlation(x, y, 2, 5)

	decode_op = decode(probs)

	with tf.Session(config=session_conf) as sess:
		with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
			#inp = [tf.constant(0.5, shape=[2, 2])] * 2
			
			cell = tf.nn.rnn_cell.BasicLSTMCell(2) # 2 is the num_units for LSTM
			enc_state = cell.zero_state(4, tf.float32) # 4 is the batch size

			#_, enc_state = tf.nn.rnn(cell, inp, dtype=tf.float32)
			#dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
			
			test_op = test(dec_inp, enc_state, cell)
			feed_dict = {dec_inp:decoder_inputs}
			feed_dict1 = {x:inp_x, y:inp_y}
			feed_dict2 = {probs: inp_probs}

			sess.run([tf.initialize_all_variables()])
			res = sess.run(test_op,feed_dict=feed_dict)
			x1, y1, corr_nr, corr_dr1, corr_dr2, corr_dr, corr = sess.run([x_op, y_op, corr_nr_op, corr_dr1_op, corr_dr2_op, corr_dr_op, corr_op],feed_dict=feed_dict1)
			z = sess.run(decode_op, feed_dict=feed_dict2)
			print corr			
			print z
			print z.shape[0]

			#print sess.run(dec_inp[0])
			#print sess.run(dec_inp[1])
			#print sess.run(dec_inp[2])
			#print len(res)
			#print res[0]
			#print res[1]
			#print res[2]
			#print res[3]
			# print('correlation %f' % (corr))
			# print x1 
			# print x1.shape
			# print y1
			# print y1.shape
			# print corr_nr 
			# print corr_nr.shape
			# print corr_dr1 
			# print corr_dr1.shape
			# print corr_dr2 
			# print corr_dr2.shape
			# print corr_dr 
			# print corr_dr.shape
			# #print res.shape
			#self.assertEqual(3, le`n(res))
			#self.assertEqual((2, 2), res[0].shape)

			#res = sess.run([mem])
			#self.assertEqual(1, len(res))
			#self.assertEqual((2, 4), res[0].shape)

# with tf.Graph().as_default():  
	
# 	W  = tf.placeholder(tf.int32, shape=(3, 2), name="W")
# 	batch_seq_inputs = np.asarray([[0,1,2],[2,1,0],[2,1,1]])
# 	w = np.asarray([[0,0],[1,1],[2,2]])

# 	feed_dict = {seq_inputs:batch_seq_inputs, W:w}

# 	train_op = train(seq_inputs, W)

# 	session_conf = tf.ConfigProto(allow_soft_placement=True)

# 	with tf.Session(config=session_conf) as sess:
# 	    	# run the op to initialize the variables
# 	    	init = tf.initialize_all_variables()
# 	    	sess.run(init)
# 	    	inputs = sess.run(train_op,feed_dict=feed_dict)
# 	    	print inputs
"""
  def compute_correlation1(self, x, y) :
    #""
    x : tensor, float32 [None, None]
    y : tensor, float32 [None, None] (same shape as x)
    :return: tensor, float32 
    correlation between x and y 
    #""
    x_mean = tf.reshape(tf.tile(tf.reduce_mean(x, 0), [self.batch_size]), [self.batch_size, self.embedding_size])
    y_mean = tf.reshape(tf.tile(tf.reduce_mean(y, 0), [self.batch_size]), [self.batch_size, self.embedding_size])

    x_centered = tf.sub(x, x_mean)
    y_centered = tf.sub(y, y_mean)

    corr_nr = tf.reduce_sum(tf.mul(x_centered, y_centered), 0)
    corr_dr1 = tf.sqrt(tf.reduce_sum(tf.mul(x_centered, x_centered), 0))
    corr_dr2 = tf.sqrt(tf.reduce_sum(tf.mul(y_centered, y_centered), 0))

    corr_dr = tf.mul(corr_dr1, corr_dr2)

    corr = tf.div(corr_nr, corr_dr)

    return x_centered, y_centered, corr_nr, corr_dr1, corr_dr2, corr_dr, tf.reduce_sum(corr)
    #return tf.reduce_sum(tf.abs(corr))
"""
