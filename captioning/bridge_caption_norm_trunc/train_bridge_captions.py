""" training script

tensorboard --logdir /home/vikasraykar/deep/exaamples/context_free_claim_sentence/exp_1/summaries/

"""

import tensorflow as tf    
import numpy as np
import argparse
import time
import datetime
import os
import json
import codecs

from vocabulary import build_vocabulary, load_vocabulary, word_ids_to_sentence, word_ids_to_sentences
from bridge_captions_model import BridgeCaptionsModel
from parallel_data_reader import ParallelDataReader
if __name__ == '__main__' :

  """
  # ==================================================
  # PARAMETERS
  # ==================================================
  """

  parser = argparse.ArgumentParser()

  # Model hyper-parameters
  parser.add_argument('--embedding_size', type=int, default=512, help='size of the common representation') 
  parser.add_argument('--fr_rnn_size', type=int, default=512, help='size of hidden state of the French RNN decoder')
  parser.add_argument('--fr_rnn_num_layers', type=int, default=1, help='number of layers in the RNN (default: 1)') 
  parser.add_argument('--model', type=str, default='gru', help='rnn, gru, basic_lstm, or lstm (default: basic_lstm)')
  parser.add_argument('--keep_prob', type=float, default=0.0, help='dropout probability')
  #use same initialization as Jana used in BC
  parser.add_argument('--init_scale_embedding', type=float, default=1.0, help='random uniform initialization in the range [-init_scale_embedding,init_scale_embedding] for the embeddign layer')
  parser.add_argument('--init_scale', type=float, default=0.1, help='random uniform initialization in the range [-init_scale,init_scale]')
  #parser.add_argument('--train_embedding_matrix', type=int, default=1, help='if 0 does not train the embedding matrix and keeps it fixed')
  #parser.add_argument('--use_pretrained_embedding_matrix', type=int, default=1, help='if 1 use the pre-trained word2vec for initializing the embedding matrix')
  #parser.add_argument('--pretrained_embedding_filename', type=str, default='/home/viraykar/deep/resources/GoogleNews-vectors-negative300.bin', help='full path to the .bin file containing the pre-trained word vectors')              
    
  # Training parameters
  parser.add_argument('--batch_size', type=int, default=80, help='batch size') # 16, 32, 64
  parser.add_argument('--max_epochs', type=int, default=50, help='number of epochs')
  parser.add_argument('--optimizer', type=str, default='adam', help='gradient_descent, adam') #rmsprop, 0.01, 0.1
  parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
  parser.add_argument('--save_dir', type=str, default='im_en_fr_01', help='directory to save checkpointed models')
  parser.add_argument('--print_every', type=int, default=1, help='print some info after these many batches')
  parser.add_argument('--summary_every', type=int, default=2000000, help='dump summaries for tensorbaord after these many batches')
  parser.add_argument('--save_every', type=int, default=20000, help='save and evaluate model after these many batches')
  parser.add_argument('--sample_every', type=int, default=500, help='prints decodes samples after these many batches')
  parser.add_argument('--valid_every', type=int, default=20000, help='runs the model on the validation data after these many batches')
  parser.add_argument('--start_from', type=str, default='', help='load existing model from this location')

  # Task specific
  parser.add_argument('--en_vocabulary', type=str, default=None, help='the path to previously computed vocabulary files for English')
  parser.add_argument('--fr_vocabulary', type=str, default=None, help='the path to previously computed vocabulary files for French')
  parser.add_argument('--vocab_min_count', type=int, default=5, help='keep words whose count is >= vocab_min_count') # 5
  parser.add_argument('--batch_mode', type=str, default='pad', help='pad, bucket : how to handle variable length sequences in a batch')
  parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
  parser.add_argument('--en_vocab_size', type=int, default=None, help='size of the English vocabulary')
  parser.add_argument('--fr_vocab_size', type=int, default=None, help='size of the French vocabulary')
  parser.add_argument('--image_size', type=int, default=4096, help='vector size of each image')
  parser.add_argument('--top_k', type=int, default=512, help='the number of top correlated columns that should be selcted')
  parser.add_argument('--max_seq_length', type=int, default=18, help='maximum sequence length allowed') # 50
  parser.add_argument('--lmbda', type=float, default=1.0, help='the scaling factor to be used for correlation') # 50
  parser.add_argument('--lmbda_cov', type=float, default=100.0, help='the scaling factor to be used for covariance between dimensions of the hidden representation') # 100
  parser.add_argument('--calculate_mean_every', type=float, default=10000, help='number of steps after which mean and stddev of h_e and h_i should be recalculated') # 50
  parser.add_argument('--pretrain_epochs', type=int, default=None, help='number of epochs for which i2e should be pretrained')
  
  parser.add_argument('--i_e_i_train', type =str, help='i portion of the i_e training data')
  parser.add_argument('--i_e_e_train', type =str, help='e portion of the i_e training data')
  parser.add_argument('--i_e_i_valid', type =str, help='i portion of the i_e valid data')
  parser.add_argument('--i_e_e_valid', type =str, help='e portion of the i_e valid data')
  
  parser.add_argument('--e_f_e_train', type =str, help='e portion of the e_f training data')
  parser.add_argument('--e_f_f_train', type =str, help='f portion of the e_f training data')
  parser.add_argument('--e_f_e_valid', type =str, help='e portion of the e_f valid data')
  parser.add_argument('--e_f_f_valid', type =str, help='f portion of the e_f valid data')
  
  parser.add_argument('--i_f_i_test', type =str, help='i portion of the i_f test data')
  parser.add_argument('--i_f_f_test', type =str, help='f portion of the i_f test data')
  args = parser.parse_args()
  
  """
  # ==================================================
  # CREATE VOCABULARY
  # ==================================================
  """
  if args.en_vocabulary != None and os.path.exists(args.en_vocabulary) :
    en_word_to_id, en_id_to_word = load_vocabulary(args.en_vocabulary, 'en')
  else :
    en_word_to_id, en_id_to_word, en_bias_vector = \
    build_vocabulary(args.i_e_e_train, min_count=args.vocab_min_count, language='en')  

  if args.en_vocabulary != None and os.path.exists(args.fr_vocabulary) :
    fr_word_to_id, fr_id_to_word = load_vocabulary(args.fr_vocabulary, 'fr')
  else :
    fr_word_to_id, fr_id_to_word, fr_bias_vector = \
    build_vocabulary(args.e_f_f_train, min_count=args.vocab_min_count, language='fr')  

  args.en_vocab_size = len(en_word_to_id)
  args.fr_vocab_size = len(fr_word_to_id)

  """
  # ==================================================
  # DATASET
  # ==================================================
  """
  parallel_data_reader = ParallelDataReader()
  
  parallel_data_reader.load_e_f_data(
    e_train_filename = args.e_f_e_train,
    f_train_filename = args.e_f_f_train,
    e_valid_filename = args.e_f_e_valid,
    f_valid_filename = args.e_f_f_valid,
    e_test_filename = args.e_f_e_valid,
    f_test_filename = args.e_f_f_valid,
    e_word_to_id = en_word_to_id, 
    e_id_to_word = en_id_to_word,
    f_word_to_id = fr_word_to_id, 
    f_id_to_word = fr_id_to_word, 
    max_sequence_length = args.max_seq_length)

  args.max_seq_length = parallel_data_reader.max_sequence_length
  
  parallel_data_reader.load_im_e_data(
    e_train_filename = args.i_e_e_train,
    im_train_filename = args.i_e_i_train, 
    e_valid_filename = args.i_e_e_valid,
    im_valid_filename = args.i_e_i_valid,
    e_test_filename = args.i_e_e_valid,
    im_test_filename = args.i_e_i_valid,
    e_word_to_id = en_word_to_id, 
    e_id_to_word = en_id_to_word,
    max_sequence_length = args.max_seq_length)

  parallel_data_reader.load_im_f_data(
    f_train_filename = args.i_f_f_test,
    im_train_filename = args.i_f_i_test,
    f_valid_filename = args.i_f_f_test,
    im_valid_filename = args.i_f_i_test,
    f_test_filename = args.i_f_f_test,
    im_test_filename = args.i_f_i_test,
    f_word_to_id = fr_word_to_id, 
    f_id_to_word = fr_id_to_word,
    max_sequence_length = args.max_seq_length)
  
  """
  # ==================================================
  # SAVE HYPERPARAMETERS AND VOCABULARY
  # ==================================================
  """
  print('----------------------------HYPERPARAMETES')
  for arg in vars(args):
    print('%s = %s'%(arg, getattr(args, arg)))

  # save the args and the vocabulary

  if not os.path.exists(args.save_dir): 
    os.makedirs(args.save_dir)  

  with open(os.path.join(args.save_dir,'args.json'),'w') as f:
    f.write(json.dumps(vars(args),indent=1))

  with open(os.path.join(args.save_dir,'en_word_to_id.json'),'w') as f:
    f.write(json.dumps(en_word_to_id,indent=1))

  with open(os.path.join(args.save_dir,'en_id_to_word.json'),'w') as f:
    f.write(json.dumps(en_id_to_word,indent=1))

  with open(os.path.join(args.save_dir,'fr_word_to_id.json'),'w') as f:
    f.write(json.dumps(fr_word_to_id,indent=1))

  with open(os.path.join(args.save_dir,'fr_id_to_word.json'),'w') as f:
    f.write(json.dumps(fr_id_to_word,indent=1))

  #args.max_seq_length = 5
  #args.embedding_size = 1024
  #args.batch_size = 10
  #args.e_vocab_size = 6
  #args.f_vocab_size = 6
  #args.image_size = 6
  #args.fr_rnn_size = 1024
  args.random_seed = 1234

  """
  # ==================================================
  # TRAINING
  # ==================================================
  """
  print('----------------------------TRAINING')
  with tf.Graph().as_default():  
    print args.max_seq_length
    model = BridgeCaptionsModel(args.max_seq_length, args.embedding_size, args.batch_size, args.fr_rnn_size, 
      args.en_vocab_size, args.fr_vocab_size, args.image_size, args.lmbda, args.random_seed,
      e_bias_init_vector=en_bias_vector, f_bias_init_vector=fr_bias_vector)
    e_sequences  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="e_seq_inputs")
    e_targets  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="e_targets")
    e_seq_masks  = tf.placeholder(tf.float32, shape=(args.batch_size, args.max_seq_length), name="e_seq_masks")
    f_sequences  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="f_seq_inputs")
    f_targets  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="f_targets")
    f_seq_masks  = tf.placeholder(tf.float32, shape=(args.batch_size, args.max_seq_length), name="f_seq_masks")
    e_bow = tf.placeholder(tf.float32, shape=(args.batch_size, args.en_vocab_size), name="e_bow")
    f_bow = tf.placeholder(tf.float32, shape=(args.batch_size, args.fr_vocab_size), name="f_bow")
    images = tf.placeholder(tf.float32, shape=(args.batch_size, args.image_size), name="images")
    do_perturb = tf.placeholder(tf.bool)
    lmbda = tf.placeholder(tf.float32)
    image1 = tf.placeholder(tf.float32, [1, args.image_size], name="sample_image")
    e_sequences1  = tf.placeholder(tf.int32, shape=(1, args.max_seq_length), name="e_seq_inputs")
    e_seq_masks1  = tf.placeholder(tf.float32, shape=(1, args.max_seq_length), name="e_seq_masks")
    e_mean = tf.placeholder(tf.float32, shape=(1, args.embedding_size), name="h_e_mean")
    e_stddev = tf.placeholder(tf.float32, shape=(1, args.embedding_size), name="h_e_stddev")
    i_mean = tf.placeholder(tf.float32, shape=(1, args.embedding_size), name="h_i_mean")
    i_stddev = tf.placeholder(tf.float32, shape=(1, args.embedding_size), name="h_i_stddev")
    ei_corr = tf.placeholder(tf.float32, shape=(1, args.embedding_size), name="ei_corr")
    ei_corr_weights = tf.placeholder(tf.float32, shape=(1, args.embedding_size), name="ei_corr_weights")
    selection_matrix = tf.placeholder(tf.float32, shape=(args.embedding_size, args.top_k), name="selection_matrix")

    in_ei_corr_weights = np.zeros((1,args.embedding_size))
    in_ei_corr_weights[:,:args.top_k] = 1.0
    in_ei_corr_weights[:,args.top_k:] = 1.0

    """
    # ==================================================
    # OPs for e-f data
    # ==================================================
    """
    h_ef_op, l_e2f_op = model.loss_e2f(e_sequences, e_seq_masks, f_sequences, f_seq_masks, e_mean, e_stddev, selection_matrix)

    train_e_f_op = model.training_e_f(l_e2f_op, args.optimizer, args.learning_rate)

    e2f_pred_seq_op = model.inference_e2f(e_sequences1, e_seq_masks1, e_mean, e_stddev, selection_matrix)
    """
    # ==================================================
    # OPs for im-e data
    # ==================================================
    """
    l_ei_corr_op, loss_ei_op, h_e_op, h_i_op, corr_vec_op = model.loss_e_i(e_sequences, e_seq_masks,
            images, lmbda, ei_corr_weights)
    
    train_e_i_op = model.training_e_i(loss_ei_op, args.optimizer, args.learning_rate)

    """
    # ==================================================
    # OPs for sampling and testing
    # ==================================================
    """
    h_if_op = model.get_standard_i(image1, i_mean, i_stddev, selection_matrix)
    i2f_pred_seq_op = model.inference_i2f(image1, i_mean, i_stddev, selection_matrix)

    """
    # ==========================================================
    # CREATE all directories for storing summaries, models, etc.
    # ==========================================================
    """
    # directory to dump ouput on test or validation data
    output_dir = os.path.abspath(os.path.join(args.save_dir, 'output'))
    if not os.path.exists(output_dir): 
      os.makedirs(output_dir)

    # directory to dump the intermediate models
    checkpoint_dir = os.path.abspath(os.path.join(args.save_dir, 'checkpoints'))
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    
    checkpoint_prefix = os.path.join(args.save_dir, 'model')  

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver(tf.all_variables())
    
    """
    # ==========================================================
    # START the training loop
    # ==========================================================
    """
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=session_conf) as sess:
      # run the op to initialize the variables
      if os.path.exists(args.start_from) :
        saver.restore(sess, args.start_from)
        print "Loading existing model from " + args.start_from
      else :
        init = tf.initialize_all_variables()
        sess.run(init)

      print('Trainable Variables')
      print '\n'.join([v.name for v in tf.trainable_variables()])

      print('All Variables')
      print '\n'.join([v.name for v in tf.all_variables()])


      in_h_e_mean = np.zeros((1, args.embedding_size))
      in_h_i_mean = np.zeros((1, args.embedding_size))
      in_h_e_stddev = np.ones((1, args.embedding_size))
      in_h_i_stddev = np.ones((1, args.embedding_size))
      step = 0
      prev_epoch = 0
      curr_epoch = 0
      corr_fp = codecs.open(output_dir + '/corr_i2e.txt', 'w')
      in_selection_matrix = np.eye(args.embedding_size)[:,:args.top_k]
      while curr_epoch <= args.max_epochs:
        curr_epoch = parallel_data_reader._epochs_completed
        step +=1
        start_time = time.time()      
        
        """
        # ===========================================================================
        # TRAIN with En-Im data for first 2 epochs and then for every alternate step
        # ===========================================================================
        """ 
        if curr_epoch <= args.pretrain_epochs or step % 2 == 1 : 
          in_e_sequences, in_e_seq_masks, in_e_targets, in_images = parallel_data_reader.next_e_i_train_batch(args.batch_size) 
          feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks, images:in_images,
                  lmbda:args.lmbda, ei_corr_weights:in_ei_corr_weights}
          l_ei_corr, loss_ei, h_e, h_i, corr_vec, _ \
            = sess.run([l_ei_corr_op, loss_ei_op, h_e_op, h_i_op, corr_vec_op, train_e_i_op], feed_dict=feed_dict)

          duration = time.time() - start_time
          
          """
          # ===========================================================================
          # PRINT En-Im loss
          # ===========================================================================
          """ 
          if step % args.print_every == 0:
            print('epoch %d batch %d: l_ei_corr = %3f loss = %3f h_i_mean = %3f h_e_mean = %3f (%.3f sec)' 
              % (curr_epoch, step, l_ei_corr, loss_ei, np.mean(h_e), np.mean(h_i), duration))

          #for vec in corr_vec :
          print >> corr_fp, 'Epoch', parallel_data_reader._epochs_completed, ' '.join([str(corr) for corr in corr_vec])

        """
        # ===================================================================================
        # TRAIN with En-Fr data. Start only after 2 epochs and then for every alternate step
        # ===================================================================================
        """ 
        if curr_epoch > args.pretrain_epochs and step % 2 == 0 : 
          in_e_sequences, in_e_seq_masks, _, in_f_sequences, in_f_seq_masks, _ = parallel_data_reader.next_e_f_train_batch(args.batch_size)  

          feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks, 
              f_sequences:in_f_sequences, f_seq_masks:in_f_seq_masks, e_mean:in_h_e_mean,
              e_stddev:in_h_e_stddev, selection_matrix: in_selection_matrix}
          
          l_e2f, _ = sess.run([l_e2f_op, train_e_f_op], feed_dict=feed_dict)

          duration = time.time() - start_time

          """
          # ===========================================================================
          # PRINT En-Fr loss
          # ===========================================================================
          """ 
          if (step - 1) % args.print_every == 0:
            print('epoch %d batch %d: l_e2f = %.3f (%.3f sec)' % (curr_epoch, step, l_e2f, duration))

          """
          # ===========================================================================
          # PRINT En-Fr samples from the training data to get a feel for the quality
          # ===========================================================================
          """ 
          if (step - 1) % args.sample_every == 0:
            print('----------------------------PRINTING SAMPLES')
            for i in range(len(in_e_sequences1)) :
              in_e_sequence = np.reshape(in_e_sequences1[i], (1, args.max_seq_length))  
              in_e_seq_mask = np.reshape(in_e_seq_masks1[i], (1, args.max_seq_length))  
              feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks,
                      e_mean:in_h_e_mean, e_stddev:in_h_e_stddev, selection_matrix: in_selection_matrix}
              e2f_pred_seq = sess.run(e2f_pred_seq_op, feed_dict=feed_dict)
              e2f_pred_seq = np.hstack(e2f_pred_seq).tolist()
              predicted = ' '.join([fr_id_to_word[x] for x in e2f_pred_seq])
              ground_truth = word_ids_to_sentences(in_e_sequences, fr_id_to_word)
              print  "Truth: " + ground_truth[i]
              print  "Predicted: " + predicted
              print

        """
        # =======================================================================================
        # CALCULATE the mean and standard deviations of h_e and h_i over the entire training data
        # =======================================================================================
        """ 
        if step % args.calculate_mean_every == 0 :
          h_e = []
          h_i = []
          index = parallel_data_reader.e_i_current_train_index
          epoch = parallel_data_reader._epochs_completed
          #reset to the first data instance 
          parallel_data_reader.e_i_current_train_index = 0
          while parallel_data_reader._epochs_completed == epoch : #go once over the entire data         
            in_e_sequences, in_e_seq_masks, in_e_targets, in_images = parallel_data_reader.next_e_i_train_batch(args.batch_size)  
            feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks, images:in_images, lmbda:args.lmbda}
            h_e_tmp, h_i_tmp = sess.run([h_e_op, h_i_op], feed_dict=feed_dict)
            h_e.extend(h_e_tmp)
            h_i.extend(h_i_tmp)

          in_h_e_mean = np.reshape(np.mean(h_e, 0), (1, args.embedding_size))
          in_h_i_mean = np.reshape(np.mean(h_i, 0), (1, args.embedding_size))
          in_h_e_stddev = np.reshape(np.std(h_e, 0), (1, args.embedding_size))
          in_h_i_stddev = np.reshape(np.std(h_i, 0), (1, args.embedding_size))
          h_e_cent = h_e - in_h_e_mean
          h_i_cent = h_i - in_h_i_mean
          numer = np.sum(h_e_cent * h_i_cent, 0)
          denom1 = np.sum(h_e_cent * h_e_cent, 0)
          denom2 = np.sum(h_i_cent * h_i_cent, 0)
          in_ei_corr = numer/np.sqrt(denom1 * denom2)
          in_ei_corr = np.reshape(in_ei_corr, (1, args.embedding_size))
          if curr_epoch == args.pretrain_epochs :
            max_corr_indices = (-in_ei_corr).argsort()[:,:args.top_k]
            print 'max-corr-indices:'  
            print max_corr_indices
            in_selection_matrix = np.eye(args.embedding_size)[:,max_corr_indices[0]]
              

          fp = codecs.open(output_dir + '/mean_stddev_' + str(step) + '.txt', 'w', 'utf-8')
          print >> fp, ' '.join([str(x) for x in in_h_e_mean[0]])
          print >> fp, ' '.join([str(x) for x in in_h_i_mean[0]])
          print >> fp, ' '.join([str(x) for x in in_h_e_stddev[0]])
          print >> fp, ' '.join([str(x) for x in in_h_i_stddev[0]])
          print >> fp, ' '.join([str(x) for x in in_ei_corr[0]])
          print >> fp, ' '.join([str(x) for x in in_ei_corr[0]])
          fp.close()
          parallel_data_reader.e_i_current_train_index = index
          parallel_data_reader._epochs_completed = epoch

        """
        # ===========================================================================
        # VALIDATE existing models on the validation set
        # ===========================================================================
        """ 

        if step % args.valid_every == 0 : #or train_data.epochs_completed == args.max_epochs:
          print('----------------------------RUNNING VALIDATION')
          fp = codecs.open(output_dir + '/valid_e2f_' + str(step) + '.txt', 'w', 'utf-8')
          fp1 = codecs.open(output_dir + '/mean_valid_e2f_' + str(step) + '.txt', 'w', 'utf-8')
          while parallel_data_reader.has_next_e_f_valid_batch(args.batch_size) :
            in_e_sequences, in_e_seq_masks, in_f_sequences, in_f_seq_masks = parallel_data_reader.next_e_f_valid_batch(args.batch_size)
            feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks, f_sequences:in_f_sequences, 
              f_seq_masks: in_f_seq_masks, e_mean:in_h_e_mean, e_stddev:in_h_e_stddev,
              selection_matrix:in_selection_matrix}
            h_ef, l_e2f = sess.run([h_ef_op, l_e2f_op], feed_dict=feed_dict)

            for i in range(len((in_e_sequences))) :
              in_e_sequence = np.reshape(in_e_sequences[i], (1, args.max_seq_length))  
              in_e_seq_mask = np.reshape(in_e_seq_masks[i], (1, args.max_seq_length))  
              feed_dict = {e_sequences1:in_e_sequence, e_seq_masks1:in_e_seq_mask,
                      e_mean:in_h_e_mean, e_stddev:in_h_e_stddev, selection_matrix:in_selection_matrix}
              e2f_pred_seq = sess.run(e2f_pred_seq_op, feed_dict=feed_dict)
              e2f_pred_seq = np.hstack(e2f_pred_seq).tolist()
              predicted = ' '.join([fr_id_to_word[x] for x in e2f_pred_seq])
              ground_truth = word_ids_to_sentences(in_f_sequences, fr_id_to_word)
              print >> fp, "Truth: " + ground_truth[i]
              print >> fp, "Predicted: " + predicted + '\n'
              print >> fp1, ' '.join([str(x) for x in h_ef[i]])
            print >> fp, "l_e2f = %3f" % (l_e2f)  
          fp.close()  
          fp1.close()

          fp = codecs.open(output_dir + '/valid_i2f_' + str(step) + '.txt', 'w', 'utf-8')
          fp1 = codecs.open(output_dir + '/mean_valid_i2f_' + str(step) + '.txt', 'w', 'utf-8')
          while parallel_data_reader.has_next_f_i_valid_batch(args.batch_size) :
            in_f_sequences, _, in_images = parallel_data_reader.next_f_i_valid_batch(args.batch_size)  
            
            for i,image in enumerate(in_images) :
              image = np.reshape(image, (1, 4096))
              feed_dict = {image1:image, i_mean:in_h_i_mean, i_stddev:in_h_i_stddev, selection_matrix: in_selection_matrix}
              h_if = sess.run(h_if_op, feed_dict=feed_dict)
              i2f_pred_seq = sess.run(i2f_pred_seq_op, feed_dict=feed_dict)
              i2f_pred_seq = np.hstack(i2f_pred_seq).tolist()
              predicted = ' '.join([fr_id_to_word[x] for x in i2f_pred_seq])
              ground_truth = word_ids_to_sentences(in_f_sequences, fr_id_to_word)
              
              print >> fp, "Valid Truth: " + ground_truth[i]
              print >> fp, "Valid Predicted: " + predicted
              print >> fp1, ' '.join([str(x) for x in h_if[0]])
          fp.close()
          fp1.close()

          
          fp = codecs.open(output_dir + '/valid_i2e_' + str(step) + '.txt', 'w', 'utf-8')
          while parallel_data_reader.has_next_e_i_valid_batch(args.batch_size) :
            in_e_sequences, in_e_seq_masks, in_images = parallel_data_reader.next_e_i_valid_batch(args.batch_size)  
            feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks, images:in_images,
                    lmbda:args.lmbda, ei_corr_weights:in_ei_corr_weights}
            l_ei_corr = sess.run(l_ei_corr_op, feed_dict=feed_dict)
            print >> fp, "Loss = %.2f" % (l_ei_corr) 
          fp.close()  
            
        # save a checkpoint 
        if step % args.save_every == 0 or curr_epoch == args.max_epochs:
          path = saver.save(sess, checkpoint_prefix, global_step=step)
          print("Saved model checkpoint to {}".format(path))








"""
    #W_en  = tf.placeholder(tf.float32, shape=(e_vocab_size, embedding_size), name="W")
    #W_fr  = tf.placeholder(tf.float32, shape=(f_vocab_size, embedding_size), name="W")
    #W_out  = tf.placeholder(tf.float32, shape=(rnn_size, f_vocab_size), name="W_out")
  
    in_e_seq_inputs = np.asarray([[1,3,4,2,0],[1,4,5,2,0],[1,4,3,5,2]]) # 0 is for PAD, 1 is for GO, 2 is for EOS and 
    in_e_seq_masks = np.asarray([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    in_f_seq_inputs = np.asarray([[1,3,4,2,0],[1,4,5,2,0],[1,4,3,5,2]]) # 0 is for PAD, 1 is for GO, 2 is for EOS and 
    in_f_seq_masks = np.asarray([[1,1,0,0,1],[0,1,1,0,1],[0,1,1,1,1]])
    in_f_targets = np.asarray([[3,4,2,0,0],[4,5,2,0,0],[4,3,5,2,0]])
    in_e_bow =  model.get_n_hot_rep(in_e_seq_inputs, args.en_vocab_size)
    in_f_bow =  model.get_n_hot_rep(in_f_seq_inputs, args.fr_vocab_size)
    #w_en = np.asarray([[0,0],[.1,.1],[.2,.2],[.3,.3],[.4,.4],[.5,.5]])
    #w_fr = np.asarray([[6,6],[7,7],[8,8],[9,9],[10,10],[11,11]])
    #w_out = np.asarray([[1,2,3,1,1,1],[3,2,1,2,3,2]])
"""


