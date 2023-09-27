import tensorflow as tf
import numpy as np
from functools import reduce
from collections import Counter

def get_data(train, test):
  with open(train) as f:
    sentences_train = f.read().split()
    # f.read().split('.\n'): split by new line
    # f.read().split(): split by space
  with open(test) as g:
    sentences_test = g.read().split()

  """
  for s in sentences_train[:10]: 
    print(s)
  print()
  for s in sentences_test[:10]: 
    print(s)
  """

  train_all = ""
  for i in sentences_train:
    train_all += " " + i
  train_all = train_all[1:]

  test_all = ""
  for i in sentences_test:
    test_all += " " + i
  test_all = test_all[1:]

  """
  print(train_all[:20])
  print()
  print(test_all[:20])
  print()
  for i in train_all[:100].split():
    print(i)
  print()
  """

  vocab = Counter(train_all.split())
  # Counter() counts the instances of each element in your list 
  # print(vocab)
  # print()

  convert_list = list(vocab)
  word2id = {w:i for i,w in enumerate(convert_list)}
  # print(word2id)
  # print()

  train2id = []
  for i in train_all.split():
    train2id.append(word2id[i])
  # print(train2id[:100])
  # print()

  test2id = []
  for i in test_all.split():
    test2id.append(word2id[i])
  # print(test2id[:100])

  return train2id, test2id, word2id
