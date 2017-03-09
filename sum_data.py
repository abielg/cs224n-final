from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin
# copied all imports from qa_data.py, for now

# need starter code to parse arguments

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2 # currently only aware that we've used this one

# assigns a unique integer to each element of the vocab (these tuples are stored in vocab)
def initialize_vocabulary(vocabulary_path): 
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


# creates and saves embedding glove matrix, where each row corresponds to a word (indices match index in vocab, for future lookup)
def process_glove(args, vocab_list, save_path, size=4e5):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        glove = np.zeros((len(vocab_list), args.glove_dim))
        not_found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size): # tqdm shows a smart progress meter
                array = line.lstrip().rstrip().split(" ") # returns array of words, split by whitespace
                word = array[0] # first element of the line is the word itself
                vector = list(map(float, array[1:])) # converts the elements of the word vectors to floats, stored in vector
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector # set idx row of glove to be the vector of floats for the word embeddings
                elif word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                elif word.lower() in vocab_list:
                    idx = vocab_list.index(word.lower())
                    glove[idx, :] = vector
                elif word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                else:
                    not_found += 1
        found = size - not_found
        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path)) # where tf does said "trimming" occur


# looks up word in vocabulary, returns ids as a list (default is UNK_ID)
def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


