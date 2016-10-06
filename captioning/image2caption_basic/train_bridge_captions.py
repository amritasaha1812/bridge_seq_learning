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
  parser.add_argument('--batch_size', type=int, default=64, help='batch size') # 16, 32, 64
  parser.add_argument('--max_epochs', type=int, default=50, help='number of epochs')
  parser.add_argument('--optimizer', type=str, default='adam', help='gradient_descent, adam') #rmsprop, 0.01, 0.1
  parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
  parser.add_argument('--save_dir', type=str, default='exp_02', help='directory to save checkpointed models')
  parser.add_argument('--print_every', type=int, default=1, help='print some info after these many batches')
  parser.add_argument('--summary_every', type=int, default=2000000, help='dump summaries for tensorbaord after these many batches')
  parser.add_argument('--save_every', type=int, default=4500, help='save and evaluate model after these many batches')
  parser.add_argument('--sample_every', type=int, default=500, help='prints decodes samples after these many batches')
  parser.add_argument('--valid_every', type=int, default=4500, help='runs the model on the validation data after these many batches')
  parser.add_argument('--start_from', type=str, default='', help='runs the model on the validation data after these many batches')

  # Task specific
  parser.add_argument('--en_vocabulary', type=str, default=None, help='the path to previously computed vocabulary files for English')
  parser.add_argument('--fr_vocabulary', type=str, default=None, help='the path to previously computed vocabulary files for French')
  parser.add_argument('--vocab_min_count', type=int, default=5, help='keep words whose count is >= vocab_min_count') # 5
  parser.add_argument('--batch_mode', type=str, default='pad', help='pad, bucket : how to handle variable length sequences in a batch')
  parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
  parser.add_argument('--en_vocab_size', type=int, default=None, help='size of the English vocabulary')
  parser.add_argument('--fr_vocab_size', type=int, default=None, help='size of the French vocabulary')
  parser.add_argument('--image_size', type=int, default=4096, help='vector size of each image')
  parser.add_argument('--max_seq_length', type=int, default=18, help='maximum sequence length allowed') # 50
  parser.add_argument('--lmbda', type=float, default=0.05, help='the scaling factor to be used for correlation') # 50

  parser.add_argument('--i_e_i_train', type =str, help='i portion of the i_e training data')
  parser.add_argument('--i_e_e_train', type =str, help='e portion of the i_e training data')
  parser.add_argument('--i_e_i_valid', type =str, help='i portion of the i_e valid data')
  parser.add_argument('--i_e_e_valid', type =str, help='e portion of the i_e valid data')
  parser.add_argument('--i_e_i_test', type =str, help='i portion of the i_e test data')
  parser.add_argument('--i_e_e_test', type =str, help='e portion of the i_e test data')

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

  """
  if args.en_vocabulary != None and os.path.exists(args.fr_vocabulary) :
    fr_word_to_id, fr_id_to_word = load_vocabulary(args.fr_vocabulary, 'fr')
  else :
    fr_word_to_id, fr_id_to_word, _ = \
    build_vocabulary('../image2caption/datasets/mscoco_train_captions.en.b64.txt', min_count=args.vocab_min_count, language='fr')  
  """
  
  args.en_vocab_size = len(en_word_to_id)
  args.fr_vocab_size = len(en_word_to_id)

  """
  # ==================================================
  # DATASET
  # ==================================================
  """
  parallel_data_reader = ParallelDataReader()
  """
  parallel_data_reader.load_e_f_data(
    e_train_filename = r'../bridge_captions/datasets/mscoco_train_captions.en.b64.txt',
    f_train_filename = r'../bridge_captions/datasets/mscoco_train_captions.fr.b64.txt', #assumes that e_filename and f_filename are sentence level parallel
    e_valid_filename = r'../bridge_captions/datasets/mscoco_valid_captions.en.txt',
    f_valid_filename = r'../bridge_captions/datasets/mscoco_valid_captions.fr.txt', #assumes that e_filename and f_filename are sentence level parallel
    e_test_filename = r'../bridge_captions/datasets/mscoco_train_captions.en.100.txt',
    f_test_filename = r'../bridge_captions/datasets/mscoco_train_captions.fr.100.txt', #assumes that e_filename and f_filename are sentence level parallel
    e_word_to_id = en_word_to_id, 
    e_id_to_word = en_id_to_word,
    f_word_to_id = fr_word_to_id, 
    f_id_to_word = fr_id_to_word, 
    max_sequence_length = args.max_seq_length)

  args.max_seq_length = parallel_data_reader.max_sequence_length
  """
  parallel_data_reader.load_im_e_data(
    e_train_filename = args.i_e_e_train,
    im_train_filename = args.i_e_i_train, 
    e_valid_filename = args.i_e_e_valid, 
    im_valid_filename = args.i_e_i_valid,
    e_test_filename = args.i_e_e_test,
    im_test_filename = args.i_e_i_test,
    e_word_to_id = en_word_to_id, 
    e_id_to_word = en_id_to_word,
    max_sequence_length = args.max_seq_length)

  """
  parallel_data_reader.load_im_f_data(
    f_train_filename = r'../bridge_captions/datasets/mscoco_train_captions.fr.b64.txt',
    im_train_filename = r'../bridge_captions/datasets/mscoco_train_images.b64.npy', #assumes that e_filename and f_filename are sentence level parallel
    f_valid_filename = r'../bridge_captions/datasets/mscoco_valid_captions.fr.txt',
    im_valid_filename = r'../bridge_captions/datasets/mscoco_valid_images.npy', #assumes that e_filename and f_filename are sentence level parallel
    f_test_filename = r'../bridge_captions/datasets/mscoco_train_captions.fr.100.txt',
    im_test_filename = r'../bridge_captions/datasets/mscoco_train_images.100.npy', #assumes that e_filename and f_filename are sentence level parallel
    f_word_to_id = fr_word_to_id, 
    f_id_to_word = fr_id_to_word,
    max_sequence_length = args.max_seq_length)
  """
  
  """
  # ==================================================
  # SAVE HYPERPARAMETERS AND VOCABULARY
  # ==================================================
  """
  if not os.path.exists(args.save_dir): 
    os.makedirs(save_dir)
  log = codecs.open(args.save_dir + '/log.txt', 'w', 'utf-8')
  print >> log, '----------------------------HYPERPARAMETES'
  for arg in vars(args):
    print >> log, '%s = %s'%(arg, getattr(args, arg))

  # save the args and the vocabulary

  if not os.path.exists(args.save_dir): 
    os.makedirs(args.save_dir)  

  with open(os.path.join(args.save_dir,'args.json'),'w') as f:
    f.write(json.dumps(vars(args),indent=1))

  with open(os.path.join(args.save_dir,'en_word_to_id.json'),'w') as f:
    f.write(json.dumps(en_word_to_id,indent=1))

  with open(os.path.join(args.save_dir,'en_id_to_word.json'),'w') as f:
    f.write(json.dumps(en_id_to_word,indent=1))
  """
  with open(os.path.join(args.save_dir,'fr_word_to_id.json'),'w') as f:
    f.write(json.dumps(fr_word_to_id,indent=1))

  with open(os.path.join(args.save_dir,'fr_id_to_word.json'),'w') as f:
    f.write(json.dumps(fr_id_to_word,indent=1))
  """
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
  print >> log, '----------------------------TRAINING'
  with tf.Graph().as_default():  
    print >> log, args.max_seq_length
    model = BridgeCaptionsModel(args.max_seq_length, args.embedding_size, args.batch_size, args.fr_rnn_size, 
      args.en_vocab_size, args.fr_vocab_size, args.image_size, args.lmbda, args.random_seed,
      bias_init_vector=en_bias_vector)
    e_sequences  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="e_seq_inputs")
    e_targets  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="e_targets")
    e_seq_masks  = tf.placeholder(tf.float32, shape=(args.batch_size, args.max_seq_length), name="e_seq_masks")
    f_sequences  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="f_seq_inputs")
    f_targets  = tf.placeholder(tf.int32, shape=(args.batch_size, args.max_seq_length), name="f_targets")
    f_seq_masks  = tf.placeholder(tf.float32, shape=(args.batch_size, args.max_seq_length), name="f_seq_masks")
    e_bow = tf.placeholder(tf.float32, shape=(args.batch_size, args.en_vocab_size), name="e_bow")
    f_bow = tf.placeholder(tf.float32, shape=(args.batch_size, args.fr_vocab_size), name="f_bow")
    images = tf.placeholder(tf.float32, shape=(args.batch_size, args.image_size), name="images")
    feed_previous = tf.placeholder(tf.bool)
    image1 = tf.placeholder(tf.float32, [1, args.image_size], name="sample_image")


    #with tf.device('/gpu:0'):
    """
    # ==================================================
    # OPs for e-f data
    # ==================================================
    """
    ##h_e_op, h_f_op, e2f_op, f2f_op, e2e_op, f2e_op, e2f_logits_op, e2f_pred_seq_op, f2f_logits_op, f2f_pred_seq_op \
    ## = model.inference_e_f(e_sequences, e_seq_masks, f_sequences, f_seq_masks, feed_previous)
    ##l_e2f_op, l_f2f_op, l_e2e_op, l_f2e_op, l_e2f_seq_op, l_f2f_seq_op, l_ef_corr_op, loss_ef_op \
    ## = model.loss_e_f(e_bow, f_bow, f_targets, h_e_op, h_f_op, e2f_op, f2f_op, e2e_op, f2e_op, e2f_logits_op, f2f_logits_op)
    #e2f_logits_op, e2f_pred_seq_op = model.inference_e_f(e_sequences, e_seq_masks, f_sequences, f_seq_masks, feed_previous)
    
    #l_e2f_seq_op = model.loss_e_f(f_targets, e2f_logits_op)

    #train_e_f_op = model.training_e_f(l_e2f_seq_op, args.optimizer, args.learning_rate)

    """
    # ==================================================
    # OPs for im-e data
    # ==================================================
    """
    #with tf.device('/gpu:1'):
    #h_ei refers to the hidden representation of English for the Image-English data
    print >> log, 'Trainable Variables'
    print >> log, '\n'.join([v.name for v in tf.trainable_variables()])

    print >> log, 'All Variables'
    print >> log, '\n'.join([v.name for v in tf.all_variables()])


    loss_ei_op = model.loss_e_i(e_sequences, e_seq_masks, images)
    
    i2e_pred_seq_op = model.build_generator(image1)
     
    train_e_i_op = model.training_e_i(loss_ei_op, args.optimizer, args.learning_rate)

    """
    # ==================================================
    # OPs for sampling and testing
    # ==================================================
    """
    #with tf.variable_scope("root") as scope:
    #_, _, _, _, _, _, _, e2f_pred_seq_op, _, f2f_pred_seq_op \
    #    = model.inference_e_f(e_sequences, e_seq_masks, f_sequences, f_seq_masks, feed_previous=True)

    #scope.reuse_variables()

    #_, i2f_pred_seq_op = model.inference_i_f(images, f_sequences, feed_previous=True)

    """
    # ==========================================================
    # STORE all losses for graph visualization using tensorboard
    # ==========================================================
    """
    # summaries for loss and accuracy
    #l_e2f_seq_summary = tf.scalar_summary('l_e2f_seq', l_e2f_seq_op)
    #l_i2e_seq_summary = tf.scalar_summary('l_i2e_seq', loss_ei_op)
    #l_e2e_seq_summary = tf.scalar_summary('l_e2e_seq', l_e2e_seq_op)
    #l_ei_corr_summary = tf.scalar_summary('l_ei_corr', l_ei_corr_op)

    #train_e_f_summary_op = tf.merge_summary([l_e2f_seq_summary])

    #train_e_i_summary_op = tf.merge_summary([l_i2e_seq_summary])

    """
    # ==========================================================
    # CREATE all directories for storing summaries, models, etc.
    # ==========================================================
    """
    # directory to dump ouput on test or validation data
    output_dir = os.path.abspath(os.path.join(args.save_dir, 'output'))
    if not os.path.exists(output_dir): 
      os.makedirs(output_dir)


    # directory to dump the summaries
    train_summary_dir = os.path.join(args.save_dir, 'summaries', 'train')
    if not os.path.exists(train_summary_dir): 
      os.makedirs(train_summary_dir)

    #valid_summary_dir = os.path.join(args.save_dir, 'summaries', 'valid')
    #if not os.path.exists(valid_summary_dir): 
    # os.makedirs(valid_summary_dir)

    # directory to dump the intermediate models
    checkpoint_dir = os.path.abspath(os.path.join(args.save_dir, 'checkpoints'))
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')  

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver(tf.all_variables())
    
    """
    # ==========================================================
    # START the training loop
    # ==========================================================
    """
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=session_conf) as sess:
      # load from previously saved checkpoint of run the init op to initialize the variables
      if os.path.exists(args.start_from) :
        saver.restore(sess, args.start_from)
        print >> log, "Loading existing model from " + args.start_from
      else :
        init = tf.initialize_all_variables()
        sess.run(init)

      
      # instantiate a SummaryWriter to output summaries and the graph
      train_summary_writer = tf.train.SummaryWriter(train_summary_dir, graph_def=sess.graph_def)

      #h_e, h_f, e2f, f2f, e2e, f2e = sess.run([h_e_op, h_f_op, e2f_op, f2f_op,
      #                            e2e_op, f2e_op], feed_dict=feed_dict)
      
      print >> log, 'Trainable Variables'
      print >> log, '\n'.join([v.name for v in tf.trainable_variables()])

      print >> log, 'All Variables'
      print >> log, '\n'.join([v.name for v in tf.all_variables()])

      step = 0
      while parallel_data_reader._epochs_completed <= args.max_epochs:
        step +=1
        start_time = time.time()      
        
        if step % 2 == 1 : #alternate between en-im and en-fr batches
          #in_e_sequences, in_e_seq_masks, in_e_targets, in_f_sequences, in_f_seq_masks, in_f_targets = parallel_data_reader.next_e_f_train_batch(args.batch_size)  
          #in_e_bow =  model.get_n_hot_rep(in_e_sequences, args.en_vocab_size)
          #in_f_bow =  model.get_n_hot_rep(in_f_sequences, args.fr_vocab_size)

          #feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks, e_targets:in_e_targets,
          # f_sequences:in_f_sequences, f_seq_masks:in_f_seq_masks, f_targets:in_f_targets, feed_previous:False}
          
          #l_e2f_seq, _ = sess.run([l_e2f_seq_op, train_e_f_op], feed_dict=feed_dict)

          duration = time.time() - start_time

          # write the summaries
          #if (step - 1) % args.summary_every == 0:
          #  summaries = sess.run(train_e_f_summary_op, feed_dict=feed_dict)
          #  train_summary_writer.add_summary(summaries, step)

          #if (step - 1) % args.print_every == 0:
          #  print('epoch %d batch %d: l_e2f_seq = %.3f (%.3f sec)' % (parallel_data_reader._epochs_completed, step, l_e2f_seq, duration))


          #if (step - 1) % args.sample_every == 0:
            #print('----------------------------PRINTING SAMPLES')
            #in_f_sequences1 = in_f_sequences
            #in_f_sequences1[:,1:] = 0
            #in_f_seq_masks1 = in_f_seq_masks
            #in_f_seq_masks1[:,1:] = 0
            #print in_f_sequences
            #print in_f_sequences1
            #feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks,
            #f_sequences:in_f_sequences, f_seq_masks:in_f_seq_masks, feed_previous:True}
            #e2f_pred_seq = sess.run(e2f_pred_seq_op, feed_dict=feed_dict)
            #predicted = word_ids_to_sentences(e2f_pred_seq, fr_id_to_word)
            #ground_truth = word_ids_to_sentences(in_f_sequences, fr_id_to_word)
            #for i in range (5) :
              #print  "Truth: " + ground_truth[i].encode('utf-8')
              #print  "Predicted: " + predicted[i].encode('utf-8')
              #print
        else :
          in_e_sequences, in_e_seq_masks, in_e_targets, in_images = parallel_data_reader.next_e_i_train_batch(args.batch_size)
          if in_e_sequences.shape[0] < args.batch_size :
            continue

          feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks, images:in_images}
          l_i2e_seq, _ \
            = sess.run([loss_ei_op, train_e_i_op], feed_dict=feed_dict)

          duration = time.time() - start_time

          # write the summaries
          #if step % args.summary_every == 0:
          #  summaries = sess.run(train_e_i_summary_op, feed_dict=feed_dict)
          #  train_summary_writer.add_summary(summaries, step)

          if step  % args.sample_every == 0:
            print >> log, '----------------------------PRINTING SAMPLES'
            #in_f_sequences1 = in_e_sequences
            #in_f_sequences1[:,1:] = 0
            #in_f_seq_masks1 = in_e_seq_masks
            #in_f_seq_masks1[:,1:] = 0
            #print in_f_sequences
            #print in_f_sequences1
            for i,image in enumerate(in_images) :
              image = np.reshape(image, (1, 4096))
              feed_dict = {image1:image}
              #f_sequences:in_f_sequences, f_seq_masks:in_f_seq_masks, feed_previous:True}
              i2e_pred_seq = sess.run(i2e_pred_seq_op, feed_dict=feed_dict)
              i2e_pred_seq = np.hstack(i2e_pred_seq).tolist()
              predicted = ' '.join([en_id_to_word[x] for x in i2e_pred_seq])
              #predicted = word_ids_to_sentence(i2e_pred_seq, en_id_to_word)
              ground_truth = word_ids_to_sentences(in_e_sequences, en_id_to_word)
              #for i in range (5) :
              print  >> log, "Truth: " + ground_truth[i]
              print  >> log, "Predicted: " + predicted
              print >> log, ""
        
          if step % args.print_every == 0:
            print >> log, 'epoch %d batch %d: loss = %.3f (%.3f sec)' %(parallel_data_reader._epochs_completed, step, l_i2e_seq, duration)

        
        if step == 0 or step % args.valid_every == 0 : #or train_data.epochs_completed == args.max_epochs:
          #print('----------------------------RUNNING VALIDATION')
          #fp = codecs.open(output_dir + '/valid_e2f_' + str(step) + '.txt', 'w', 'utf-8')
          #while parallel_data_reader.has_next_e_f_valid_batch(args.batch_size) :
          #  in_e_sequences, in_e_seq_masks, in_f_sequences1, in_f_seq_masks1 = parallel_data_reader.next_e_f_valid_batch(args.batch_size)
          #  in_f_sequences1[:,1:] = 0
          #  in_f_seq_masks1[:,1:] = 0  
          #  feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks,f_sequences:in_f_sequences1,
          #   f_seq_masks:in_f_seq_masks1, feed_previous:True}
          #  e2f_pred_seq = sess.run(e2f_pred_seq_op, feed_dict=feed_dict)
          #  predicted = word_ids_to_sentences(e2f_pred_seq, fr_id_to_word)
          #  ground_truth = word_ids_to_sentences(in_f_sequences1, fr_id_to_word)
          #  for i in range(len(predicted)) :
          #    print >> fp, "Truth: " + ground_truth[i]
          #    print >> fp, "Predicted: " + predicted[i] + '\n'
          #fp.close()  
          
          fp = codecs.open(output_dir + '/valid_i2e_' + str(step) + '.txt', 'w', 'utf-8')
          while parallel_data_reader.has_next_e_i_valid_batch(args.batch_size) :
            in_e_sequences, in_e_seq_masks, in_images = parallel_data_reader.next_e_i_valid_batch(args.batch_size)
            feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks, images:in_images}  
            l_i2e_seq_val = sess.run(loss_ei_op, feed_dict=feed_dict)

            for i,image in enumerate(in_images) :
              if (i % 1) == 0 :   
                image = np.reshape(image, (1, 4096))
                feed_dict = {image1:image}
                #f_sequences:in_f_sequences, f_seq_masks:in_f_seq_masks, feed_previous:True}
                i2e_pred_seq = sess.run(i2e_pred_seq_op, feed_dict=feed_dict)
                i2e_pred_seq = np.hstack(i2e_pred_seq).tolist()
                predicted = ' '.join([en_id_to_word[x] for x in i2e_pred_seq])
                #predicted = word_ids_to_sentence(i2e_pred_seq, en_id_to_word)
                ground_truth = word_ids_to_sentences(in_e_sequences, en_id_to_word)
                #for i in range (5) :
                print >> fp, "Valid Truth: " + ground_truth[i]
                print >> fp, "Valid Predicted: " + predicted
            print >> fp, "Loss: %.3f \n" % (l_i2e_seq_val)
          fp.close()
          
          fp = codecs.open(output_dir + '/test_i2e_' + str(step) + '.txt', 'w', 'utf-8')
          while parallel_data_reader.has_next_e_i_test_batch(args.batch_size) :
            in_e_sequences, in_e_seq_masks, in_images = parallel_data_reader.next_e_i_test_batch(args.batch_size)
            feed_dict = {e_sequences:in_e_sequences, e_seq_masks:in_e_seq_masks, images:in_images}  
            l_i2e_seq_val = sess.run(loss_ei_op, feed_dict=feed_dict)

            for i,image in enumerate(in_images) :
              if (i % 1) == 0 :   
                image = np.reshape(image, (1, 4096))
                feed_dict = {image1:image}
                #f_sequences:in_f_sequences, f_seq_masks:in_f_seq_masks, feed_previous:True}
                i2e_pred_seq = sess.run(i2e_pred_seq_op, feed_dict=feed_dict)
                i2e_pred_seq = np.hstack(i2e_pred_seq).tolist()
                predicted = ' '.join([en_id_to_word[x] for x in i2e_pred_seq])
                #predicted = word_ids_to_sentence(i2e_pred_seq, en_id_to_word)
                ground_truth = word_ids_to_sentences(in_e_sequences, en_id_to_word)
                #for i in range (5) :
                print >> fp, "Valid Truth: " + ground_truth[i]
                print >> fp, "Valid Predicted: " + predicted
            print >> fp, "Loss: %.3f \n" % (l_i2e_seq_val)
          fp.close()
          #fp = codecs.open(output_dir + '/valid_i2f_' + str(step) + '.txt', 'w', 'utf-8')
          #while parallel_data_reader.has_next_f_i_valid_batch(args.batch_size) :
          #  in_f_sequences, _, in_images = parallel_data_reader.next_f_i_valid_batch(args.batch_size)  
          #  feed_dict = {f_sequences:in_f_sequences, images:in_images}
          #  i2f_pred_seq = sess.run(i2f_pred_seq_op, feed_dict=feed_dict)
          #  predicted = word_ids_to_sentences(i2f_pred_seq, fr_id_to_word)
          #  ground_truth = word_ids_to_sentences(in_f_sequences, fr_id_to_word)
            
          #  for i in range(len(predicted)) :
          #    print >> fp, "Truth: " + ground_truth[i]
          #    print >> fp, "Predicted: " + predicted[i] + '\n'
          #fp.close()
        # save a checkpoint and evaluate the model on the entire data
        if step % args.save_every == 0 : #or train_data.epochs_completed == args.max_epochs:
          # save the model
          path = saver.save(sess, checkpoint_prefix, global_step=step)
          print >> log, "Saved model checkpoint to {}".format(path)


















          """
          # evaluate 
          accuracy,true_count,num_examples = evaluate_model(sess,
            train_data,
            seq_inputs,
            seq_lengths,
            labels,
            accuracy_op,
            args.batch_size)

          print('---------------------------------------train accuracy : %0.04f [%d/%d]' %(accuracy,true_count,num_examples))

          accuracy,true_count,num_examples = evaluate_model(sess,
            valid_data,
            seq_inputs,
            seq_lengths,
            labels,
            accuracy_op,
            args.batch_size)
          print('---------------------------------------valid accuracy : %0.04f [%d/%d]' %(accuracy,true_count,num_examples))
          """ 

"""          
        if epoch % 1000 == 0:
          e2f_pred_seq, l_e2f_seq \
            = sess.run([e2f_pred_seq_op, l_e2f_seq_op], feed_dict=feed_dict)
        
          print e2f_pred_seq
          print in_f_bow
          print l_e2f_seq

    x_op, y_op, corr_nr_op, corr_dr1_op, corr_dr2_op, corr_dr_op, corr_op = model.compute_correlation1(h_e_op, h_i_op)

                    h_e, h_i, x1, y1, corr_nr, corr_dr1, corr_dr2, corr_dr, corr = sess.run([h_e_op, h_i_op, x_op, y_op, corr_nr_op, corr_dr1_op, corr_dr2_op, corr_dr_op, corr_op],feed_dict=feed_dict)
          print('correlation %f' % (corr))
          print h_e
          print h_i
          print x1 
          print x1.shape
          print y1
          print y1.shape
          print corr_nr 
          print corr_nr.shape
          print corr_dr1 
          print corr_dr1.shape
          print corr_dr2 
          print corr_dr2.shape
          print corr_dr 
          print corr_dr.shape

"""
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


