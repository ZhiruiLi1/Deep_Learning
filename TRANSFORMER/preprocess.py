import numpy as np
import tensorflow as tf
import numpy as np

from attenvis import AttentionVis
av = AttentionVis()

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
FRENCH_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 14
##########DO NOT CHANGE#####################

def pad_corpus(french, english):
	"""
	DO NOT CHANGE:
	arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
	text is given an initial "*STOP*". English is padded with "*START*" at the beginning for Teacher Forcing.
	:param french: list of French sentences
	:param french: list of French sentences
	:param english: list of English sentences
	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
	"""
	FRENCH_padded_sentences = []
	for line in french:
		padded_FRENCH = line[:FRENCH_WINDOW_SIZE]
		padded_FRENCH += [STOP_TOKEN] + [PAD_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_FRENCH)-1)
		FRENCH_padded_sentences.append(padded_FRENCH)

	ENGLISH_padded_sentences = []
	for line in english:
		padded_ENGLISH = line[:ENGLISH_WINDOW_SIZE]
		padded_ENGLISH = [START_TOKEN] + padded_ENGLISH + [STOP_TOKEN] + [PAD_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_ENGLISH)-1)
		ENGLISH_padded_sentences.append(padded_ENGLISH)

	return FRENCH_padded_sentences, ENGLISH_padded_sentences

def build_vocab(sentences):
	"""
	DO NOT CHANGE
  Builds vocab from list of sentences
	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE
  Convert sentences to indexed
	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
	DO NOT CHANGE
  Load text data from file
	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text


def get_data(french_training_file, english_training_file, french_test_file, english_test_file):
  """
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.
	:param french_training_file: Path to the French training file.
	:param english_training_file: Path to the English training file.
	:param french_test_file: Path to the French test file.
	:param english_test_file: Path to the English test file.

	:return: Tuple of train containing:
	(2-d list or array with English training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with English test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with French training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with French test sentences in vectorized/id form [num_sentences x 14]),
	English vocab (Dict containg word->index mapping),
	French vocab (Dict containg word->index mapping),
	English padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""


  # MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index

	#TODO:
	
  #1) Read English and French Data for training and testing (see read_data)
  french_train = read_data(french_training_file)
  french_test = read_data(french_test_file)
  english_train = read_data(english_training_file)
  english_test = read_data(english_test_file)

  print(english_train[0])

	#2) Pad training data (see pad_corpus)
  pad_french_train, pad_english_train = pad_corpus(french_train, english_train)
  
  print(pad_english_train[0])  # length 15, one additional item *START*
  print(pad_french_train[0])   # length 14

	#3) Pad testing data (see pad_corpus)
  pad_french_test, pad_english_test = pad_corpus(french_test, english_test)

	#4) Build vocab for French (see build_vocab)
  vocab_french, vocab_french_pad = build_vocab(pad_french_train)
	#5) Build vocab for English (see build_vocab)   # DICTIONARY 
  vocab_english, vocab_english_pad = build_vocab(pad_english_train)

  print(vocab_english['*START*']) # 42
  print(vocab_english['edited']) # 3079
  print(vocab_english['hansard']) # 3968

	#6) Convert training and testing English sentences to list of IDS (see convert_to_id)
  id_english_train = convert_to_id(vocab_english, pad_english_train)
  id_english_test = convert_to_id(vocab_english, pad_english_test)

  print(id_english_train[0])

	#7) Convert training and testing French sentences to list of IDS (see convert_to_id)
  id_french_train = convert_to_id(vocab_french, pad_french_train)
  id_french_test = convert_to_id(vocab_french, pad_french_test)

  """
  print("this is id_english_train")
  print(id_english_train)
  print("this is id_english_test")
  print(id_english_test)
  print("this is id_french_train")
  print(id_french_train)
  print("this is id_french_test")
  print(id_french_test)
  print("this is vocab_english")
  print(vocab_english)
  print("this is vocab_french")
  print(vocab_french)
  print("this is vocab_english_pad")
  print(vocab_english_pad)
  """

  print(len(vocab_english))
  print(len(vocab_french))
  
  return id_english_train, id_english_test, id_french_train, id_french_test, vocab_english, vocab_french, vocab_english_pad

# get_data("fls.txt", "els.txt", "flt.txt", "elt.txt")
