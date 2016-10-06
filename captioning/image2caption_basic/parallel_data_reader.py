
import codecs, sys, os
from vocabulary import build_vocabulary, load_vocabulary
import numpy as np

class ParallelDataReader():

	def sentence_to_word_ids(self, sentence, word_to_id, max_sequence_length = None):
		""" encode a given [sentence] to a list of word ids using the vocabulary dict [word_to_id]
		adds a end-of-sentence marker (<EOS>) out-of-vocabulary words are mapped to 3   
		"""
		sentence = "<GO> " + sentence
		tokens = sentence.strip().split(' ')

		if max_sequence_length is not None:
		    tokens = tokens[:max_sequence_length-1]

		tokens.append('<EOS>')
		#print tokens
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

	def load_e_f_data(self, e_train_filename, f_train_filename, e_valid_filename, f_valid_filename,
	 e_test_filename, f_test_filename, e_word_to_id, e_id_to_word, f_word_to_id, f_id_to_word, max_sequence_length) :

		self.e_train_sequences, self.e_train_seq_lengths, self.e_train_seq_masks, self.e_f_num_examples, self.max_sequence_length \
			= self.get_sequences(e_train_filename, e_word_to_id, max_sequence_length, True)
		self.e_valid_sequences, self.e_valid_seq_lengths, self.e_valid_seq_masks, self.e_valid_num_examples, _ \
			= self.get_sequences(e_valid_filename, e_word_to_id, self.max_sequence_length)
		self.e_test_sequences, self.e_test_seq_lengths, self.e_test_seq_masks, self.e_test_num_examples, _ \
			= self.get_sequences(e_test_filename, e_word_to_id, self.max_sequence_length)

		self.f_train_sequences, self.f_train_seq_lengths, self.f_train_seq_masks, self.f_train_num_examples, self.f_max_sequence_length \
			= self.get_sequences(f_train_filename, f_word_to_id, self.max_sequence_length)
		self.f_valid_sequences, self.f_valid_seq_lengths, self.f_valid_seq_masks, self.f_valid_num_examples, _ \
			= self.get_sequences(f_valid_filename, f_word_to_id, self.max_sequence_length)
		self.f_test_sequences, self.f_test_seq_lengths, self.f_test_seq_masks, self.f_test_num_examples, _ \
			= self.get_sequences(f_test_filename, f_word_to_id, self.max_sequence_length)

		self.f_train_targets = np.roll(self.f_train_sequences, -1)
		self.f_train_targets[:, -1] = 0

		self.e_train_targets = np.roll(self.e_train_sequences, -1)
		self.e_train_targets[:, -1] = 0

		self._epochs_completed = 0
		self.e_f_current_train_index = 0
		self.e_f_current_valid_index = 0
		self.e_f_current_test_index = 0

	def load_im_e_data(self, e_train_filename, im_train_filename, e_valid_filename, im_valid_filename,
	 e_test_filename, im_test_filename, e_word_to_id, e_id_to_word, max_sequence_length) :

		self.ei_train_sequences, self.ei_train_seq_lengths, self.ei_train_seq_masks, self.ei_num_examples, self.max_sequence_length  \
			= self.get_sequences(e_train_filename, e_word_to_id, max_sequence_length)
		self.ei_valid_sequences, self.ei_valid_seq_lengths, self.ei_valid_seq_masks, self.ei_valid_num_examples, _ \
			= self.get_sequences(e_valid_filename, e_word_to_id, self.max_sequence_length)
		self.ei_test_sequences, self.ei_test_seq_lengths, self.ei_test_seq_masks, self.ei_test_num_examples, _ \
			= self.get_sequences(e_test_filename, e_word_to_id, self.max_sequence_length)

		self.ei_train_targets = np.roll(self.ei_train_sequences, -1)
		self.ei_train_targets[:, -1] = 0

		self.im_train = np.load(im_train_filename)
		self.im_valid = np.load(im_valid_filename)
		self.im_test = np.load(im_test_filename)
		
		self._epochs_completed = 0
		self.e_i_current_train_index = 0
		self.e_i_current_valid_index = 0
		self.e_i_current_test_index = 0

	def load_im_f_data(self, f_train_filename, im_train_filename, f_valid_filename, im_valid_filename,
	 f_test_filename, im_test_filename, f_word_to_id, f_id_to_word, max_sequence_length) :

		self.fi_train_sequences, self.fi_train_seq_lengths, self.fi_train_seq_masks, self.fi_num_examples, _ \
			= self.get_sequences(f_train_filename, f_word_to_id, max_sequence_length)
		self.fi_valid_sequences, self.fi_valid_seq_lengths, self.fi_valid_seq_masks, self.fi_valid_num_examples, _ \
			= self.get_sequences(f_valid_filename, f_word_to_id, self.max_sequence_length)
		self.fi_test_sequences, self.fi_test_seq_lengths, self.fi_test_seq_masks, self.fi_test_num_examples, _ \
			= self.get_sequences(f_test_filename, f_word_to_id, self.max_sequence_length)

		self.imf_train = np.load(im_train_filename)
		self.imf_valid = np.load(im_valid_filename)
		self.imf_test = np.load(im_test_filename)
		
		self.f_i_current_train_index = 0
		self.f_i_current_valid_index = 0
		self.f_i_current_test_index = 0

		# print e_word_to_id
		# print 
		# print f_word_to_id
		# print 
		# print self.e_train_sequences
		# print 
		# print self.e_train_seq_masks
		# print
		# print self.f_train_sequences
		# print 
		# print self.f_train_seq_masks

	def next_e_f_train_batch(self, batch_size):
		""" return the next [batch_size] examples from this data set
		:params:
		batch_size: int
    the batch size 

		:returns:
		inputs: np.int32 - [batch_size, seq_length]
		labels: np.int32 - [batch_size]
		seq_lengths: np.int32 - [batch_size]        
		"""
		start = self.e_f_current_train_index
		  
		self.e_f_current_train_index += batch_size
      
		end = min(self.e_f_current_train_index, self.e_f_num_examples)

		e_sequences = self.e_train_sequences[start:end]
		f_sequences = self.f_train_sequences[start:end]

		e_seq_masks = self.e_train_seq_masks[start:end]
		f_seq_masks = self.f_train_seq_masks[start:end]

		f_targets = self.f_train_targets[start:end]
		e_targets = self.e_train_targets[start:end]

		if self.e_f_current_train_index >= self.e_f_num_examples:
			# finished eopch
			self._epochs_completed += 1
			# Shuffle the data
			perm = list(np.arange(self.e_f_num_examples))
			#np.random.shuffle(perm)
			self.e_train_sequences = self.e_train_sequences[perm]
			self.f_train_sequences = self.f_train_sequences[perm]
			self.e_train_seq_masks = self.e_train_seq_masks[perm]
			self.f_train_seq_masks = self.f_train_seq_masks[perm]
			self.f_train_targets = self.f_train_targets[perm]
			# Start next epoch
			self.e_f_current_train_index = 0

			#print('start = %d end =  %d epoch = %d'%(start,end,self._epochs_completed)) 

		return e_sequences, e_seq_masks, e_targets, f_sequences, f_seq_masks, f_targets        

	def next_e_i_train_batch(self, batch_size):
		""" return the next [batch_size] examples from this data set
		:params:
		batch_size: int
    		the batch size 

		"""
		start = self.e_i_current_train_index
		  
		self.e_i_current_train_index += batch_size
      
		end = min(self.e_i_current_train_index, self.ei_num_examples)

		e_sequences = self.ei_train_sequences[start:end]
		e_targets = self.ei_train_targets[start:end]
		images = self.im_train[start:end]

		e_seq_masks = self.ei_train_seq_masks[start:end]
		if self.e_i_current_train_index >= self.ei_num_examples:
			# finished eopch
			self._epochs_completed += 1
			# Shuffle the data
			#perm = list(np.arange(self.ei_num_examples))
			#np.random.shuffle(perm)
			#self.ei_train_sequences = self.ei_train_sequences[perm]
			#self.images = self.im_train[perm]
			#self.ei_train_seq_masks = self.ei_train_seq_masks[perm]
			# Start next epoch
			self.e_i_current_train_index = 0

			#print('start = %d end =  %d epoch = %d'%(start,end,self._epochs_completed)) 

		return e_sequences, e_seq_masks, e_targets, images


	def has_next_e_i_valid_batch(self, batch_size, reset=True):
		if self.e_i_current_valid_index + batch_size > self.ei_valid_num_examples:
			if reset :
				self.e_i_current_valid_index = 0
			return False
		return True		


	def next_e_i_valid_batch(self, batch_size):
		""" return the next [batch_size] examples from this data set
		:params:
		batch_size: int
    		the batch size 

		"""
		start = self.e_i_current_valid_index
		  
		self.e_i_current_valid_index += batch_size
      
		end = min(self.e_i_current_valid_index, self.ei_valid_num_examples)

		e_sequences = self.ei_valid_sequences[start:end]
		images = self.im_valid[start:end]

		e_seq_masks = self.ei_valid_seq_masks[start:end]


		return e_sequences, e_seq_masks, images

	def has_next_e_f_valid_batch(self, batch_size, reset=True):
		if self.e_f_current_valid_index + batch_size > self.f_valid_num_examples:
			if reset :
				self.e_f_current_valid_index = 0
			return False
		return True		


	def next_e_f_valid_batch(self, batch_size):
		""" return the next [batch_size] examples from this data set
		:params:
		batch_size: int
    		the batch size 

		"""
		start = self.e_f_current_valid_index
		  
		self.e_f_current_valid_index += batch_size
      
		end = min(self.e_f_current_valid_index, self.f_valid_num_examples)

		e_sequences = self.e_valid_sequences[start:end]
		f_sequences = self.f_valid_sequences[start:end]

		e_seq_masks = self.e_valid_seq_masks[start:end]
		f_seq_masks = self.f_valid_seq_masks[start:end]


		return e_sequences, e_seq_masks, f_sequences, f_seq_masks


	def has_next_f_i_valid_batch(self, batch_size, reset=True):
		if self.f_i_current_valid_index + batch_size > self.fi_valid_num_examples:
			if reset :
				self.f_i_current_valid_index = 0
			return False
		return True		


	def next_f_i_valid_batch(self, batch_size):
		""" return the next [batch_size] examples from this data set
		:params:
		batch_size: int
    		the batch size 

		"""
		start = self.f_i_current_valid_index
		  
		self.f_i_current_valid_index += batch_size
      
		end = min(self.f_i_current_valid_index, self.fi_valid_num_examples)

		f_sequences = self.fi_valid_sequences[start:end]
		images = self.imf_valid[start:end]

		f_seq_masks = self.fi_valid_seq_masks[start:end]

		return f_sequences, f_seq_masks, images


	def has_next_e_i_test_batch(self, batch_size, reset=True):
		if self.e_i_current_test_index + batch_size > self.ei_test_num_examples:
			if reset :
				self.e_i_current_test_index = 0
			return False
		return True		


	def next_e_i_test_batch(self, batch_size):
		""" return the next [batch_size] examples from this data set
		:params:
		batch_size: int
    		the batch size 

		"""
		start = self.e_i_current_test_index
		  
		self.e_i_current_test_index += batch_size
      
		end = min(self.e_i_current_test_index, self.ei_test_num_examples)

		e_sequences = self.ei_test_sequences[start:end]
		images = self.im_test[start:end]

		e_seq_masks = self.ei_test_seq_masks[start:end]


		return e_sequences, e_seq_masks, images


if __name__ == '__main__':
	e_word_to_id, e_id_to_word = build_vocabulary('datasets/mscoco_train_captions.en.10K.txt', min_count=1, language='en')  
	f_word_to_id, f_id_to_word = build_vocabulary('datasets/mscoco_train_captions.fr.10K.txt', min_count=1, language='fr')  
	max_sequence_length = 100
	parallel_data_reader = ParallelDataReader()

	parallel_data_reader.load_e_f_data(
    e_train_filename = r'datasets/mscoco_train_captions.en.10K.txt',
    f_train_filename = r'datasets/mscoco_train_captions.fr.10K.txt', #assumes that e_filename and f_filename are sentence level parallel
    e_valid_filename = r'datasets/mscoco_train_captions.en.10K.txt',
    f_valid_filename = r'datasets/mscoco_train_captions.fr.10K.txt', #assumes that e_filename and f_filename are sentence level parallel
    e_test_filename = r'datasets/mscoco_train_captions.en.10K.txt',
    f_test_filename = r'datasets/mscoco_train_captions.fr.10K.txt', #assumes that e_filename and f_filename are sentence level parallel
    e_word_to_id = en_word_to_id, 
    e_id_to_word = en_id_to_word,
    f_word_to_id = fr_word_to_id, 
    f_id_to_word = fr_id_to_word, 
    max_sequence_length = args.max_seq_length)

	parallel_data_reader.load_im_e_data(
    e_train_filename = r'datasets/mscoco_train_captions.en.10K.txt',
    im_train_filename = r'datasets/mscoco_train_images.10K.npy', #assumes that e_filename and f_filename are sentence level parallel
    e_valid_filename = r'datasets/mscoco_train_captions.en.10K.txt',
    im_valid_filename = r'datasets/mscoco_train_images.10K.npy', #assumes that e_filename and f_filename are sentence level parallel
    e_test_filename = r'datasets/mscoco_train_captions.en.10K.txt',
    im_test_filename = r'datasets/mscoco_train_images.10K.npy', #assumes that e_filename and f_filename are sentence level parallel
    e_word_to_id = en_word_to_id, 
    e_id_to_word = en_id_to_word,
    max_sequence_length = args.max_seq_length)

	print e_f_num_examples, e_f_current_train_index

	print ei_num_examples, ei_num_examples
