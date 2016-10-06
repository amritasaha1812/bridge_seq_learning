
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

	def get_sequences(self, train_filename, word_to_id, max_sequence_length) :
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

		sequences = np.zeros([num_examples, max_sequence_length], dtype=np.int32)
		for idx,s in enumerate(input_sentences):
		  sequences[idx,:seq_lengths[idx]] = s

		sequence_masks = np.zeros([num_examples, max_sequence_length], dtype=np.int32)
		for idx,s in enumerate(input_sentences):
		  sequence_masks[idx,:seq_lengths[idx]] = 1

		fp.close()
		return sequences, seq_lengths, sequence_masks, num_examples, max_sequence_length

	def load_data(self, e_train_filename, f_train_filename, e_valid_filename, f_valid_filename,
	 e_test_filename, f_test_filename, e_word_to_id, e_id_to_word, f_word_to_id, f_id_to_word, max_sequence_length) :

		self.e_train_sequences, self.e_train_seq_lengths, self.e_train_seq_masks, self.e_f_num_examples, self.max_sequence_length \
			= self.get_sequences(e_train_filename, e_word_to_id, max_sequence_length)
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

		#self.f_train_targets = np.roll(self.f_train_sequences, -1)
		#self.f_train_targets[:, -1] = 0

		#self.e_train_targets = np.roll(self.e_train_sequences, -1)
		#self.e_train_targets[:, -1] = 0
		perm = list(np.arange(self.e_f_num_examples))
		np.random.shuffle(perm)
		self.e_train_sequences = self.e_train_sequences[perm]
		self.f_train_sequences = self.f_train_sequences[perm]
		self.e_train_seq_lengths = self.e_train_seq_lengths[perm]
		self.f_train_seq_lengths = self.f_train_seq_lengths[perm]
		self.e_train_seq_masks = self.e_train_seq_masks[perm]
		self.f_train_seq_masks = self.f_train_seq_masks[perm]

		self._epochs_completed = 0
		self._current_train_index = 0
		self._current_valid_index = 0
		self._current_test_index = 0

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
		start = self._current_train_index
		  
		self._current_train_index += batch_size
      
		end = min(self._current_train_index, self.e_f_num_examples)

		e_sequences = self.e_train_sequences[start:end]
		f_sequences = self.f_train_sequences[start:end]

		e_seq_lengths = self.e_train_seq_lengths[start:end]
		f_seq_lengths = self.f_train_seq_lengths[start:end]

		e_seq_masks = self.e_train_seq_masks[start:end]
		f_seq_masks = self.f_train_seq_masks[start:end]

		if self._current_train_index >= self.e_f_num_examples:
			# finished eopch
			self._epochs_completed += 1
			# Shuffle the data
			perm = list(np.arange(self.e_f_num_examples))
			#np.random.shuffle(perm)
			self.e_train_sequences = self.e_train_sequences[perm]
			self.f_train_sequences = self.f_train_sequences[perm]
			self.e_train_seq_lengths = self.e_train_seq_lengths[perm]
			self.f_train_seq_lengths = self.f_train_seq_lengths[perm]
			self.e_train_seq_masks = self.e_train_seq_masks[perm]
			self.f_train_seq_masks = self.f_train_seq_masks[perm]
			# Start next epoch
			self._current_train_index = 0

			#print('start = %d end =  %d epoch = %d'%(start,end,self._epochs_completed)) 

		return e_sequences, e_seq_lengths, e_seq_masks, f_sequences, f_seq_lengths, f_seq_masks

	def has_next_valid_batch(self, batch_size, reset=True):
		if self._current_valid_index + batch_size > self.f_valid_num_examples:
			if reset :
				self._current_valid_index = 0
			return False
		return True		

	def next_valid_batch(self, batch_size):
		""" return the next [batch_size] examples from this data set
		:params:
		batch_size: int
    		the batch size 
		"""
		start = self._current_valid_index
		  
		self._current_valid_index += batch_size
      
		end = min(self._current_valid_index, self.f_valid_num_examples)

		e_sequences = self.e_valid_sequences[start:end]
		f_sequences = self.f_valid_sequences[start:end]

		e_seq_lengths = self.e_valid_seq_lengths[start:end]
		f_seq_lengths = self.f_valid_seq_lengths[start:end]
		
                e_seq_masks = self.e_valid_seq_masks[start:end]
		f_seq_masks = self.f_valid_seq_masks[start:end]

		return e_sequences, e_seq_lengths, e_seq_masks, f_sequences, f_seq_lengths, f_seq_masks


if __name__ == '__main__':
	e_word_to_id, e_id_to_word, _ = build_vocabulary('datasets/train/en-ka-hi/en-hi.en', min_count=1, language='en')  
	f_word_to_id, f_id_to_word, _ = build_vocabulary('datasets/train/en-ka-hi/en-hi.hi', min_count=1, language='fr')  
	max_seq_length = 18

        e_f_reader = ParallelDataReader()
        e_f_reader.load_data(
            e_train_filename = r'../bridge_transliteration/datasets/train/en-ka-hi/en-hi.en',
            f_train_filename = r'../bridge_transliteration/datasets/train/en-ka-hi/en-hi.hi', 
            e_valid_filename = r'../bridge_transliteration/datasets/test/en-ka-hi/en-hi.en',
            f_valid_filename = r'../bridge_transliteration/datasets/test/en-ka-hi/en-hi.hi', #assumes
            e_test_filename = r'../bridge_transliteration/datasets/test/en-ka-hi/en-hi.en',
            f_test_filename = r'../bridge_transliteration/datasets/test/en-ka-hi/en-hi.hi', #assumes
            e_word_to_id = e_word_to_id, 
            e_id_to_word = e_id_to_word,
            f_word_to_id = f_word_to_id, 
            f_id_to_word = f_id_to_word, 
            max_sequence_length = max_seq_length)

        for i in range(10):
            e_seq, e_seq_lengths, e_seq_masks, f_seq, f_seq_lengths, f_seq_masks = e_f_reader.next_train_batch(1)

            print e_seq
            print e_seq_lengths
            print f_seq
            print f_seq_lengths
            print f_seq_masks


