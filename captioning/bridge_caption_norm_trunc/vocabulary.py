""" tools to build vocabulary
"""

#from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import codecs
import os, json

__author__  = "Vikas Raykar"
__email__   = "viraykar@in.ibm.com"

__all__ = ["build_vocabulary","sentence_to_word_ids","word_ids_to_sentence"]

def sentence_to_word_ids(sentence, word_to_id, max_sequence_length = None):
    """ encode a given [sentence] to a list of word ids using the vocabulary dict [word_to_id]
    adds a end-of-sentence marker (<EOS>)
    out-of-vocabulary words are mapped to 2    
    """
    tokens = sentence.split(' ')

    if max_sequence_length is not None:
        tokens = tokens[:max_sequence_length-1]

    tokens.append('<EOS>')

    return [word_to_id.get(word,2) for word in tokens]

def word_ids_to_sentence(word_ids_list, id_to_word):
    """ decode a given list of word ids [word_ids_list] to a sentence using the inverse vocabulary dict [id_to_word]
    """
    tokens = [id_to_word.get(id) for id in word_ids_list if id >= 2]

    return ' '.join(tokens).capitalize()+'.'

def word_ids_to_sentences(list_of_list_of_word_ids, id_to_word):
    """ decode a given list of word ids [word_ids_list] to a sentence using the inverse vocabulary dict [id_to_word]
    """
    #the 0-th dimension corresponds to sentences and 1st dimension corresponds to word_ids within the sentence
    sentences = []
    for i in range(list_of_list_of_word_ids.shape[0]) :
        tokens = [id_to_word.get(id) for id in list_of_list_of_word_ids[i] if id >= 2]
        sentences.append(' '.join(tokens))

    return sentences


def load_vocabulary(vocabulary_dir, language='en'):
    with codecs.open(os.path.join(vocabulary_dir, language + '_word_to_id.json'), 'r', 'utf-8') as f:
        word_to_id = json.load(f)
    with codecs.open(os.path.join(vocabulary_dir, language + '_id_to_word.json'), 'r', 'utf-8') as f:
        id_to_word = json.load(f)
    print('%s vocabulary size = %d'%(language, len(word_to_id))) 
    return word_to_id, id_to_word

def build_vocabulary(input_file, min_count = 5, language='en'):
    """ build the vocabulary from all the sentences in 
    the  input file

    :params:
        input_file: string
            path to the input file
        min_count: int
            keep words whose count is >= min_count                 

    :returns:
       word_to_id: dict
            dict mapping a word to its id, e.g., word_to_id['the'] = 4
            the id start from 4
            3 is reserved for out-of-vocabulary words (<OOV>)
            2 is reserved for end-of-sentence marker (<EOS>)
            1 is reserved for begin-of-sentence marker (<GO>)
            0 is reserved for padding (<PAD>)
    """
    num_sentences = 0
    wordcount = Counter()
    fp = codecs.open(input_file, 'r', 'utf-8')
    for sentence in fp:
        tokens = sentence.strip().split(' ')
        wordcount.update(tokens)
        num_sentences += 1

    print('%s vocabulary size = %d'%(language, len(wordcount))) 

    # filtering
    count_pairs = wordcount.most_common()
    count_pairs = [c for c in count_pairs if c[1] >= min_count]

    word_counts = {}
    for c in count_pairs :
        word_counts[c[0]] = c[1]

    word_counts['<PAD>'] = num_sentences
    word_counts['<GO>'] = num_sentences
    word_counts['<EOS>'] = num_sentences
    word_counts['<OOV>'] = num_sentences

    words, _ = list(zip(*count_pairs))

    word_to_id = dict(zip(words, range(4,len(words)+4)))
    print('%s vocabulary size = %d (after filtering with min_count =  %d)'%(language, len(word_to_id),min_count)) 

    word_to_id['<PAD>'] = 0
    word_to_id['<GO>'] = 1
    word_to_id['<EOS>'] = 2
    word_to_id['<OOV>'] = 3

    id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))

    #print id_to_word

    bias_init_vector = np.array([1.0*word_counts[id_to_word[i]] for i in id_to_word])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    #print bias_init_vector
    return word_to_id, id_to_word, bias_init_vector

if __name__ == '__main__' :
    build_vocabulary('../bridge_captions/datasets/mscoco_train_captions.en.1K.txt', min_count=5, language='en')
