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

def convert_to_int(d1, d2) :
  x = {}
  y = {}
  for k in d1.keys():
    x[k] = int(d1[k])
  
  for k in d2.keys():
    y[int(k)] = d2[k]

  
  return x, y

if __name__ == '__main__' :

  """
  # ==================================================
  # PARAMETERS
  # ==================================================
  """

  parser = argparse.ArgumentParser()

  parser.add_argument('--config', type=str, default='', help='load existing model from this location')
  parser.add_argument('--model', type=str, default='', help='load existing model from this location')
  parser.add_argument('--e_test_file', type =str, help='f portion of the e_f test data')
  parser.add_argument('--f_true_file', type =str, help='f portion of the e_f test data')
  parser.add_argument('--f_out_file', type=str, default='', help='load existing model from this location')
  parser.add_argument('--vocabulary_dir', type=str, default='', help='load existing model from this location')
  
  args = parser.parse_args()
  
  args1 = json.load(codecs.open(args.config, 'r', 'utf-8'))
  print args1
  args.max_seq_length = args1['max_seq_length']
  args.embedding_size = args1['embedding_size']
  args.rnn_size = args1['rnn_size']
  args.batch_size = args1['batch_size']
  """
  # ==================================================
  # LOAD VOCABULARY
  # ==================================================
  """

  f_word_to_id, f_id_to_word = load_vocabulary(args.vocabulary_dir, 'f')
  e_word_to_id, e_id_to_word = load_vocabulary(args.vocabulary_dir, 'e')

  f_word_to_id, f_id_to_word = convert_to_int(f_word_to_id, f_id_to_word)
  e_word_to_id, e_id_to_word = convert_to_int(e_word_to_id, e_id_to_word)

  args.e_vocab_size = len(e_word_to_id)
  args.f_vocab_size = len(f_word_to_id)

  print f_id_to_word
  """
  # ==================================================
  # DATASET
  # ==================================================
  """
  e_f_reader = ParallelDataReader()
  e_f_reader.load_data(
    e_train_filename = args.e_test_file,
    f_train_filename = args.f_true_file,
    e_valid_filename = args.e_test_file,
    f_valid_filename = args.f_true_file,
    e_test_filename = args.e_test_file,
    f_test_filename = args.f_true_file,
    e_word_to_id = e_word_to_id, 
    e_id_to_word = e_id_to_word,
    f_word_to_id = f_word_to_id, 
    f_id_to_word = f_id_to_word, 
    max_sequence_length = args.max_seq_length)

  with tf.Graph().as_default():  
    model = TransliterationModel(args.max_seq_length, args.embedding_size, args.batch_size, args.rnn_size,  
      args.e_vocab_size, args.f_vocab_size, 1234)
    
    #e validation/test inputs
    e_sequences1  = tf.placeholder(tf.int32, shape=(1, args.max_seq_length), name="e_seq_inputs_test")
    e_seq_masks1  = tf.placeholder(tf.float32, shape=(1, args.max_seq_length), name="e_seq_masks_test")
    e_seq_lengths1  = tf.placeholder(tf.float32, shape=(1), name="e_seq_lengths_test")

    """
    # ==================================================
    # OPs for e-f data
    # ==================================================
    """

    e2f_pred_seq_op = model.inference_e2f1(e_sequences1, e_seq_lengths1)

    saver = tf.train.Saver(tf.all_variables())
    """
    # ==========================================================
    # START the training loop
    # ==========================================================
    """
    args.batch_size = 1
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=session_conf) as sess:
      # run the op to initialize the variables
      saver.restore(sess, args.model)

      total_count = 0
      correct_count = 0
      fp = codecs.open(args.f_out_file, 'w', 'utf-8')
      while e_f_reader.has_next_valid_batch(args.batch_size) :
        in_e_sequences, in_e_seq_lengths, _, in_f_sequences, _, in_f_seq_masks = e_f_reader.next_valid_batch(args.batch_size)

        for i in range(len((in_e_sequences))) :
          in_e_sequence = np.reshape(in_e_sequences[i], (1, args.max_seq_length))  
          in_e_seq_length = np.reshape(in_e_seq_lengths[i], (1))  
          feed_dict = {e_sequences1:in_e_sequence, e_seq_lengths1:in_e_seq_length}
          e2f_pred_seq = sess.run(e2f_pred_seq_op, feed_dict=feed_dict)
          e2f_pred_seq = np.hstack(e2f_pred_seq).tolist()
          predicted = ' '.join([f_id_to_word[x] for x in e2f_pred_seq])
          ground_truth = word_ids_to_sentences(in_f_sequences, f_id_to_word)
          print >> fp, "Truth: " + ground_truth[i].split('<EOS>')[0]
          print >> fp, "Predicted: " + predicted.split('<EOS>')[0] + '\n'
          total_count += 1
          if ground_truth[i].split('<EOS>')[0].strip() == predicted.split('<EOS>')[0].strip() :
            correct_count += 1
      print "correct = %d total = %d ACC-1 = %3f" % (correct_count, total_count, float(correct_count)/float(total_count))
      fp.close()  


