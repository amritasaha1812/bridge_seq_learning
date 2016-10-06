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
from transliteration_model import TransliterationModel
from parallel_data_reader import ParallelDataReader
if __name__ == '__main__' :

  """
  # ==================================================
  # PARAMETERS
  # ==================================================
  """

  parser = argparse.ArgumentParser()

  # Model hyper-parameters
  parser.add_argument('--embedding_size', type=int, default=64, help='size of the common representation') 
  parser.add_argument('--rnn_size', type=int, default=64, help='size of hidden state of the French RNN decoder')
  parser.add_argument('--rnn_num_layers', type=int, default=1, help='number of layers in the RNN (default: 1)') 
  parser.add_argument('--model', type=str, default='basic_lstm', help='rnn, gru, basic_lstm, or lstm (default: basic_lstm)')
  parser.add_argument('--keep_prob', type=float, default=0.0, help='dropout probability')
    
  # Training parameters
  parser.add_argument('--batch_size', type=int, default=16, help='batch size') # 16, 32, 64
  parser.add_argument('--max_epochs', type=int, default=50, help='number of epochs')
  parser.add_argument('--optimizer', type=str, default='adam', help='gradient_descent, adam') #rmsprop, 0.01, 0.1
  parser.add_argument('--learning_rate', type=float, default=1e-1, help='learning rate')
  parser.add_argument('--save_dir', type=str, default='exp_01', help='directory to save checkpointed models')
  parser.add_argument('--print_every', type=int, default=1, help='print some info after these many batches')
  parser.add_argument('--summary_every', type=int, default=2000000, help='dump summaries for tensorbaord after these many batches')
  parser.add_argument('--save_every', type=int, default=600, help='save and evaluate model after these many batches')
  parser.add_argument('--sample_every', type=int, default=1000, help='prints decodes samples after these many batches')
  parser.add_argument('--valid_every', type=int, default=1000, help='runs the model on the validation data after these many batches')
  parser.add_argument('--start_from', type=str, default='', help='load existing model from this location')

  # Task specific
  parser.add_argument('--e_vocabulary', type=str, default=None, help='the path to previously computed vocabulary files for Pivot language')
  parser.add_argument('--f_vocabulary', type=str, default=None, help='the path to previously computed vocabulary files for Target language')
  parser.add_argument('--e_vocab_size', type=int, default=None, help='size of the pivot language vocabulary')
  parser.add_argument('--f_vocab_size', type=int, default=None, help='size of the target language vocabulary')
  parser.add_argument('--vocab_min_count', type=int, default=1, help='keep words whose count is >= vocab_min_count') # 5
  parser.add_argument('--max_seq_length', type=int, default=18, help='maximum sequence length allowed') # 50
  parser.add_argument('--e_f_e_train', type =str, help='e portion of the e_f training data')
  parser.add_argument('--e_f_f_train', type =str, help='f portion of the e_f training data')
  parser.add_argument('--e_f_e_valid', type =str, help='e portion of the e_f valid data')
  parser.add_argument('--e_f_f_valid', type =str, help='f portion of the e_f valid data')
  parser.add_argument('--e_f_e_test', type =str, help='e portion of the e_f test data')
  parser.add_argument('--e_f_f_test', type =str, help='f portion of the e_f test data')
  
  args = parser.parse_args()
  

  """
  # ==================================================
  # CREATE VOCABULARY
  # ==================================================
  """
  if args.e_vocabulary != None and os.path.exists(args.e_vocabulary) :
    e_word_to_id, e_id_to_word = load_vocabulary(args.e_vocabulary, 'e')
  else :
    e_word_to_id, e_id_to_word, e_bias_vector = \
    build_vocabulary(args.e_f_e_train, min_count=args.vocab_min_count, language='f')  

  if args.f_vocabulary != None and os.path.exists(args.f_vocabulary) :
    f_word_to_id, f_id_to_word = load_vocabulary(args.f_vocabulary, 'f')
  else :
    f_word_to_id, f_id_to_word, f_bias_vector = \
    build_vocabulary( args.e_f_f_train, min_count=args.vocab_min_count, language='f')  

  args.e_vocab_size = len(e_word_to_id)
  args.f_vocab_size = len(f_word_to_id)

  """
  # ==================================================
  # DATASET
  # ==================================================
  """
  e_f_reader = ParallelDataReader()
  e_f_reader.load_data(
    e_train_filename = args.e_f_e_train,
    f_train_filename = args.e_f_f_train,
    e_valid_filename = args.e_f_e_valid,
    f_valid_filename = args.e_f_f_valid,
    e_test_filename = args.e_f_e_test,
    f_test_filename = args.e_f_f_test,
    e_word_to_id = e_word_to_id, 
    e_id_to_word = e_id_to_word,
    f_word_to_id = f_word_to_id, 
    f_id_to_word = f_id_to_word, 
    max_sequence_length = args.max_seq_length)

  # directory to dump ouput on test or validation data
  output_dir = os.path.abspath(os.path.join(args.save_dir, 'output'))
  if not os.path.exists(output_dir): 
    os.makedirs(output_dir)
  log = codecs.open(output_dir + '/log.txt', 'w', 'utf-8')
  """
  # ==================================================
  # SAVE HYPERPARAMETERS AND VOCABULARY
  # ==================================================
  """
  print >> log, '----------------------------HYPERPARAMETES'
  for arg in vars(args):
    print >> log, '%s = %s'%(arg, getattr(args, arg))

  # save the args and the vocabulary

  if not os.path.exists(args.save_dir): 
    os.makedirs(args.save_dir)  

  with open(os.path.join(args.save_dir,'args.json'),'w') as f:
    f.write(json.dumps(vars(args),indent=1))

  with open(os.path.join(args.save_dir,'e_word_to_id.json'),'w') as f:
    f.write(json.dumps(e_word_to_id,indent=1))

  with open(os.path.join(args.save_dir,'e_id_to_word.json'),'w') as f:
    f.write(json.dumps(e_id_to_word,indent=1))

  with open(os.path.join(args.save_dir,'f_word_to_id.json'),'w') as f:
    f.write(json.dumps(f_word_to_id,indent=1))

  with open(os.path.join(args.save_dir,'f_id_to_word.json'),'w') as f:
    f.write(json.dumps(f_id_to_word,indent=1))

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
  print >> log, "----------------------------TRAINING"
  with tf.Graph().as_default():  
    print >> log, args.max_seq_length
    model = TransliterationModel(args.max_seq_length, args.embedding_size, args.batch_size, args.rnn_size, 
      args.e_vocab_size, args.f_vocab_size, args.random_seed, e_bias_init_vector=e_bias_vector, f_bias_init_vector=f_bias_vector)
    
    #e inputs
    e_sequences  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="e_seq_inputs")
    e_targets  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="e_targets")
    e_seq_masks  = tf.placeholder(tf.float32, shape=(args.batch_size, args.max_seq_length), name="e_seq_masks")
    e_seq_lengths  = tf.placeholder(tf.float32, shape=(args.batch_size), name="e_seq_lengths")

    #f inputs
    f_sequences  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="f_seq_inputs")
    f_targets  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="f_targets")
    f_seq_masks  = tf.placeholder(tf.float32, shape=(args.batch_size, args.max_seq_length), name="f_seq_masks")

    #e validation/test inputs
    e_sequences1  = tf.placeholder(tf.int32, shape=(1, args.max_seq_length), name="e_seq_inputs_test")
    e_seq_masks1  = tf.placeholder(tf.float32, shape=(1, args.max_seq_length), name="e_seq_masks_test")
    e_seq_lengths1  = tf.placeholder(tf.float32, shape=(1), name="e_seq_lengths_test")

    lr = tf.placeholder(tf.float32, name="lr")

    """
    # ==================================================
    # OPs for e-f data
    # ==================================================
    """

    h_ef_op, l_e2f_op = model.loss_e2f(e_sequences, e_seq_lengths, f_sequences, f_seq_masks)

    train_e_f_op = model.training_e_f(l_e2f_op, args.optimizer, lr)

    e2f_pred_seq_op = model.inference_e2f(e_sequences1, e_seq_lengths1)

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
    
    e2f_acc_fp = codecs.open(output_dir + '/e2f_acc.txt', 'w', 'utf-8')
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
        print >> log, "Loading existing model from " + args.start_from
      else :
        init = tf.initialize_all_variables()
        sess.run(init)

      print >> log, 'Trainable Variables'
      print >> log, '\n'.join([v.name for v in tf.trainable_variables()])

      print >> log, 'All Variables'
      print >> log, '\n'.join([v.name for v in tf.all_variables()])


      step = 0
      prev_epoch = 0
      curr_epoch = 0
      while curr_epoch <= args.max_epochs:
        curr_epoch = e_f_reader._epochs_completed
        step +=1
        start_time = time.time()      

        """
        # ===================================================================================
        # TRAIN with En-Fr data. Start only after 2 epochs and then for every alternate step
        # ===================================================================================
        """ 
        if step % 2 == 0 : 
          in_e_sequences, in_e_seq_lengths, _, in_f_sequences, _, in_f_seq_masks = e_f_reader.next_train_batch(args.batch_size)  
          if in_e_sequences.shape[0] < args.batch_size :
            continue

          feed_dict = {e_sequences:in_e_sequences, e_seq_lengths:in_e_seq_lengths, 
              f_sequences:in_f_sequences, f_seq_masks:in_f_seq_masks, lr:args.learning_rate}
          
          h_ef, l_e2f, _ = sess.run([h_ef_op, l_e2f_op, train_e_f_op], feed_dict=feed_dict)

          duration = time.time() - start_time
          if curr_epoch == 0 :
            fp = codecs.open(output_dir + '/h_e_' + str(curr_epoch) + '.txt', 'a', 'utf-8')
            for i in range(len(h_ef)):
              print >> fp, ' '.join([str(x) for x in h_ef[i]])
            fp.close()


          """
          # ===========================================================================
          # PRINT En-Fr loss
          # ===========================================================================
          """ 
          if (step - 1) % args.print_every == 0:
            print >> log, 'epoch %d batch %d: l_e2f = %.3f (%.3f sec)' % (curr_epoch, step, l_e2f, duration)

          """
          # ===========================================================================
          # PRINT En-Fr samples from the training data to get a feel for the quality
          # ===========================================================================
          """ 
          if step  % args.sample_every == 0:
            print >> log, '----------------------------PRINTING SAMPLES'
            fp = codecs.open(output_dir + '/sample_e2f_' + str(step) + '.txt', 'w', 'utf-8')
            for i in range(len(in_e_sequences)) :
              in_e_sequence = np.reshape(in_e_sequences[i], (1, args.max_seq_length))  
              in_e_seq_length = np.reshape(in_e_seq_lengths[i], (1))  
              feed_dict = {e_sequences1:in_e_sequence, e_seq_lengths1:in_e_seq_length}
              e2f_pred_seq = sess.run(e2f_pred_seq_op, feed_dict=feed_dict)
              e2f_pred_seq = np.hstack(e2f_pred_seq).tolist()
              predicted = ' '.join([f_id_to_word[x] for x in e2f_pred_seq])
              ground_truth = word_ids_to_sentences(in_f_sequences, f_id_to_word)
              print >> fp, "Truth: " + ground_truth[i]
              print >> fp, "Predicted: " + predicted
              print
            fp.close()  

        """
        # ===========================================================================
        # VALIDATE existing models on the validation set
        # ===========================================================================
        """ 
        if step % args.valid_every == 0 : #or train_data.epochs_completed == args.max_epochs:
          print >> log, '----------------------------RUNNING VALIDATION'
          total_count = 0
          correct_count = 0
          fp = codecs.open(output_dir + '/valid_e2f_' + str(step) + '.txt', 'w', 'utf-8')
          fp1 = codecs.open(output_dir + '/mean_valid_e2f_' + str(step) + '.txt', 'w', 'utf-8')
          fp2 = codecs.open(output_dir + '/debug_valid_e2f_' + str(step) + '.txt', 'w', 'utf-8')
          while e_f_reader.has_next_valid_batch(args.batch_size) :
            in_e_sequences, in_e_seq_lengths, _, in_f_sequences, _, in_f_seq_masks = e_f_reader.next_valid_batch(args.batch_size)
            feed_dict = {e_sequences:in_e_sequences, e_seq_lengths:in_e_seq_lengths, f_sequences:in_f_sequences, f_seq_masks: in_f_seq_masks}
            h_ef, l_e2f = sess.run([h_ef_op, l_e2f_op], feed_dict=feed_dict)

            for i in range(len((in_e_sequences))) :
              print >>fp2, ' '.join([str(x) for x in in_e_sequences[i]]), '****',in_e_seq_lengths[i]  
              in_e_sequence = np.reshape(in_e_sequences[i], (1, args.max_seq_length))  
              in_e_seq_length = np.reshape(in_e_seq_lengths[i], (1))  
              feed_dict = {e_sequences1:in_e_sequence, e_seq_lengths1:in_e_seq_length}
              e2f_pred_seq = sess.run(e2f_pred_seq_op, feed_dict=feed_dict)
              e2f_pred_seq = np.hstack(e2f_pred_seq).tolist()
              predicted = ' '.join([f_id_to_word[x] for x in e2f_pred_seq])
              ground_truth = word_ids_to_sentences(in_f_sequences, f_id_to_word)
              print >> fp, "Truth: " + ground_truth[i].split('<EOS>')[0]
              print >> fp, "Predicted: " + predicted.split('<EOS>')[0] + '\n'
              print >> fp1, ' '.join([str(x) for x in h_ef[i]])
              total_count += 1
              if ground_truth[i].split('<EOS>')[0].strip() == predicted.split('<EOS>')[0].strip() :
                  correct_count += 1
            print >> fp, "l_e2f = %3f" % (l_e2f)  
          print >> e2f_acc_fp, "Epoch = %d step = %d correct = %d total = %d ACC-1 = %3f" % (curr_epoch, step, correct_count, total_count, float(correct_count)/float(total_count))
          fp.close()  
          fp1.close()
          fp2.close()

        # save a checkpoint 
        if step % args.valid_every == 0 :
          path = saver.save(sess, checkpoint_prefix, global_step=step)
          print >> log, "Saved model checkpoint to {}".format(path)

    log.close()
    e2f_acc_fp.close()



# h_ef_op, l_e2f_op = model.loss_e2f(e_sequences, e_seq_lengths, f_sequences, f_seq_masks, e_mean, e_stddev, selection_matrix)
# train_e_f_op = model.training_e_f(l_e2f_op, args.optimizer, args.learning_rate)
# e2f_pred_seq_op = model.inference_e2f(e_sequences1, e_seq_lengths1, e_mean, e_stddev, selection_matrix)
# l_eg_corr_op, loss_eg_op, h_e_op, h_g_op, corr_vec_op = model.loss_g2e(e_sequences, e_seq_lengths, g_sequences, g_seq_lengths, lmbda, eg_corr_weights)
# train_e_g_op = model.training_e_g(loss_eg_op, args.optimizer, args.learning_rate)
# g2f_pred_seq_op = model.inference_g2f(g_sequences1, g_seq_lengths1, g_mean, g_stddev, selection_matrix)





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


