import codecs, sys, os
from vocabulary import build_vocabulary, load_vocabulary
import numpy as np

class ImageCaptionReader():

	def sentence_to_word_ids(self, sentence, word_to_id, max_sequence_length = None):
		""" encode a given [sentence] to a list of word ids using the vocabulary dict [word_to_id]
		adds a end-of-sentence marker (<EOS>) out-of-vocabulary words are mapped to 3   
		"""
		sentence = "<GO> " + sentence
		tokens = sentence.strip().split(' ')

		if max_sequence_length is not None:
		    tokens = tokens[:max_sequence_length-1]

		tokens.append('<EOS>')
		print tokens
		return [word_to_id.get(word, 3) for word in tokens]

	def get_sequences(self, train_filename, word_to_id, max_sequence_length, recompute_max_seq_length=False) :
		""" read the training file and encode the sentences to a list of word ids using the vocabulary dict [word_to_id]
		adds a end-of-sentence marker (<EOS>) out-of-vocabulary words are mapped to 3   
		"""
		fp = codecs.open(train_filename, 'r', 'utf-8')
		#fp = open(train_filename)

		input_sentences = []
		for sentence in fp: 
		  tokens = self.sentence_to_word_ids(sentence, word_to_id = word_to_id, max_sequence_length = max_sequence_length)
		  input_sentences.append(tokens)  

		num_examples = len(input_sentences) 
		seq_lengths = np.array([len(s) for s in input_sentences],dtype=np.int32)
		if recompute_max_seq_length :    
			max_sequence_length = int(max(seq_lengths))  

		sequences = np.zeros([num_examples, max_sequence_length], dtype=np.int32)
		for idx,s in enumerate(input_sentences):
		  sequences[idx,:seq_lengths[idx]] = s

		sequence_masks = np.zeros([num_examples, max_sequence_length], dtype=np.int32)
		for idx,s in enumerate(input_sentences):
		  sequence_masks[idx,:seq_lengths[idx]] = 1

		fp.close()
		return sequences, seq_lengths, sequence_masks, num_examples, max_sequence_length

	def __init__(self, e_train_filename, im_train_filename, e_valid_filename, im_valid_filename,
	 e_test_filename, im_test_filename, e_word_to_id, e_id_to_word, max_sequence_length) :

		self.e_train_sequences, self.e_train_seq_lengths, self.e_train_seq_masks, self._num_examples, self.max_sequence_length \
			= self.get_sequences(e_train_filename, e_word_to_id, max_sequence_length, True)
		self.e_valid_sequences, self.e_valid_seq_lengths, self.e_valid_seq_masks, self.e_valid_num_examples, _ \
			= self.get_sequences(e_valid_filename, e_word_to_id, self.max_sequence_length)
		self.e_test_sequences, self.e_test_seq_lengths, self.e_test_seq_masks, self.e_test_num_examples, _ \
			= self.get_sequences(e_test_filename, e_word_to_id, self.max_sequence_length)

		self.im_train = 
		self.im_valid = 
		self.im_test = 
		
		self._epochs_completed = 0
		self._train_index_in_epoch = 0
		self._valid_index_in_epoch = 0
		self._test_index_in_epoch = 0

		print e_word_to_id
		print 
		print f_word_to_id
		print 
		print self.e_train_sequences
		print 
		print self.e_train_seq_masks
		print
		print self.f_train_sequences
		print 
		print self.f_train_seq_masks

	def next_train_batch(self, batch_size):
		""" return the next [batch_size] examples from this data set
		:params:
		batch_size: int
    the batch size 

		:returns:
		inputs: np.int32 - [batch_size, seq_length]
		labels: np.int32 - [batch_size]
		seq_lengths: np.int32 - [batch_size]        
		"""
		start = self._train_index_in_epoch
		  
		self._train_index_in_epoch += batch_size
      
		end = min(self._train_index_in_epoch, self._num_examples)

		e_sequences = self.e_train_sequences[start:end]
		f_sequences = self.f_train_sequences[start:end]

		e_seq_masks = self.e_train_seq_masks[start:end]
		f_seq_masks = self.f_train_seq_masks[start:end]

		f_targets = self.f_targets[start:end]

		if self._train_index_in_epoch >= self._num_examples:
			# finished eopch
			self._epochs_completed += 1
			# Shuffle the data
			perm = list(np.arange(self._num_examples))
			#np.random.shuffle(perm)
			self.e_train_sequences = self.e_train_sequences[perm]
			self.f_train_sequences = self.f_train_sequences[perm]
			self.e_train_seq_masks = self.e_train_seq_masks[perm]
			self.f_train_seq_masks = self.f_train_seq_masks[perm]
			self.f_targets = self.f_targets[perm]
			# Start next epoch
			self._train_index_in_epoch = 0

			#print('start = %d end =  %d epoch = %d'%(start,end,self._epochs_completed)) 

		return e_sequences, e_seq_masks, f_sequences, f_seq_masks, f_targets        

if __name__ == '__main__':
	e_word_to_id, e_id_to_word = build_vocabulary('datasets/mscoco_train_captions.en.3.txt', min_count=1, language='en')  
	f_word_to_id, f_id_to_word = build_vocabulary('datasets/mscoco_train_captions.fr.3.txt', min_count=1, language='fr')  
	max_sequence_length = 100
	parallel_data_reader = ParallelDataReader(
		e_train_filename = r'datasets/mscoco_train_captions.en.3.txt',
		f_train_filename = r'datasets/mscoco_train_captions.fr.3.txt', #assumes that e_filename and f_filename are sentence level parallel
		e_valid_filename = r'datasets/mscoco_train_captions.en.3.txt',
		f_valid_filename = r'datasets/mscoco_train_captions.fr.3.txt', #assumes that e_filename and f_filename are sentence level parallel
		e_test_filename = r'datasets/mscoco_train_captions.en.3.txt',
		f_test_filename = r'datasets/mscoco_train_captions.fr.3.txt', #assumes that e_filename and f_filename are sentence level parallel
		e_word_to_id = e_word_to_id, 
		e_id_to_word = e_id_to_word,
		f_word_to_id = f_word_to_id, 
		f_id_to_word = f_id_to_word,
		max_sequence_length = max_sequence_length)



