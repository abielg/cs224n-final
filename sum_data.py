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
_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2 # currently only aware that we've used this one


def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "summarization")
    glove_dir = os.path.join("download", "dwr")
    source_dir = os.path.join("data", "summarization")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=200, type=int) #check this default value
    parser.add_argument("--vocab_size", default=10000, type=int)
    return parser.parse_args()

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]

# assigns a unique integer to each element of the vocab (these tuples are stored in vocab)
def initialize_vocabulary(vocabulary_path, vocab_size): 
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab[:vocab_size])])
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


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


# looks up word in vocabulary, returns ids as a list (default is UNK_ID)
def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path, vocab_size,
                      tokenizer=None):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path, vocab_size)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

if __name__ == '__main__':
    args = setup_args()
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    train_path = pjoin(args.source_dir, "train")
    valid_path = pjoin(args.source_dir, "val")
    test_path = pjoin(args.source_dir, "test")

    create_vocabulary(vocab_path,
                      [pjoin(args.source_dir, "train.title"),
                       pjoin(args.source_dir, "train.article"),
                       pjoin(args.source_dir, "val.title"),
                       pjoin(args.source_dir, "val.article")])

    print("NOTE: Vocab size is set to %d" % args.vocab_size)
    vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"), args.vocab_size)

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim))

    # ======== Creating Dataset =========
    # We created our data files seperately
    # If your model loads data differently (like in bulk)
    # You should change the below code


    x_train_dis_path = train_path + ".ids.article"
    y_train_ids_path = train_path + ".ids.title"
    data_to_token_ids(train_path + ".article", x_train_dis_path, vocab_path, args.vocab_size)
    data_to_token_ids(train_path + ".title", y_train_ids_path, vocab_path, args.vocab_size)

    x_dis_path = valid_path + ".ids.article"
    y_ids_path = valid_path + ".ids.title"
    data_to_token_ids(valid_path + ".article", x_dis_path, vocab_path, args.vocab_size)
    data_to_token_ids(valid_path + ".title", y_ids_path, vocab_path, args.vocab_size)
    '''
    x_dis_path = test_path + ".ids.article"
    y_ids_path = test_path + ".ids.title"
    data_to_token_ids(test_path + ".article", x_dis_path, vocab_path, args.vocab_size)
    data_to_token_ids(test_path + ".title", y_ids_path, vocab_path, args.vocab_size)
    '''