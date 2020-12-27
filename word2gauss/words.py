
from itertools import islice

import numpy as np

from embeddings import text_to_pairs

LARGEST_UINT32 = 4294967295

def tokenizer(s):
    '''
    Whitespace tokenizer
    '''
    return s.strip().split()

class Vocabulary(object):
    '''
    Implemetation of the Vocabulary interface

    .word2id: given a token, return the id or raise KeyError if not in the vocab
    .id2word: given a token id, return the token or raise IndexError if invalid
    .tokenize: given a string, tokenize it using the tokenizer and then
    remove all OOV tokens
    .tokenize_ids: given a string, tokenize and return the token ids
    .random_ids: given an integer, return a numpy array of random token ids
    '''


    def __init__(self, tokens, tokenizer=tokenizer):
        '''
        tokens: a {'token1': 0, 'token2': 1, ...} map of token -> id
            the ids must run from 0 to n_tokens - 1 inclusive
        tokenizer: accepts a string and returns a list of strings
        '''
        self._tokens = tokens
        self._ids = {i: token for token, i in tokens.items()}
        self._ntokens = len(tokens)
        self._tokenizer = tokenizer

    def word2id(self, word):
        return self._tokens[word]

    def id2word(self, i):
        try:
            return self._ids[i]
        except KeyError:
            raise IndexError

    def tokenize(self, s):
        '''
        Removes OOV tokens using built
        '''
        tokens = self._tokenizer(s)
        return [token for token in tokens if token in self._tokens]

    def tokenize_ids(self, s, remove_oov=True):
        tokens = self._tokenizer(s)
        if remove_oov:
            return np.array([self.word2id(token)
                                for token in tokens if token in self._tokens],
                                dtype=np.uint32)

        else:
            ret = np.zeros(len(tokens), dtype=np.uint32)
            for k, token in enumerate(tokens):
                try:
                    ret[k] = self.word2id(token)
                except KeyError:
                    ret[k] = LARGEST_UINT32
            return ret

    def random_ids(self, num):
        return np.random.randint(0, self._ntokens, size=num).astype(np.uint32)


def iter_pairs(fin, vocab, dynamic_window_size=True, nsamples=2, batch_size=1, window=5):
    '''
    Convert a document stream to batches of pairs used for training embeddings.

    iter_pairs is a generator that yields batches of pairs that can
    be passed to GaussianEmbedding.train

    fin = an iterator of documents / sentences (e.g. a file like object)
        Each element is a string of raw text

    vocab = something implementing the Vocabulary interface

    dynamic_window_size = Boolean with True meaning window adapts to list length
                          and False indicating it is a constant length

    nsamples = number of negative samples to drawn for each center word

    if dynamic_window_size == False then can additionally specify

        batch_size = size of batches

        window = Number of words to the left and right of center word to include
                 as positive pairs

    '''
    documents = iter(fin)

    if dynamic_window_size == True:
        batch_size = 1
        batch = list(islice(documents, batch_size))
        while len(batch) > 0:
            #print(batch)
            text = [
                vocab.tokenize_ids(related_entity_lists, remove_oov=False)
                for related_entity_lists in batch
            ]
            #print(text)
            #print("text shape = {}".format(text[0].shape))
            dynamic_window_size = text[0].shape[0]
            #print("dynamic_window_size = {}".format(dynamic_window_size))
            #num_training_examples = dynamic_window_size * (dynamic_window_size - 1) * nsamples
            #print("num_training_examples = {}".format(num_training_examples))
            pairs = text_to_pairs(text, vocab.random_ids,
                nsamples_per_word=nsamples,
                half_window_size=dynamic_window_size)
            #print("pairs shape = {}".format(pairs.shape))
            #for i in range(pairs.shape[0]):
                #print(pairs[i,:])
            yield pairs
            batch = list(islice(documents, batch_size))

    elif dynamic_window_size == False:
        batch = list(islice(documents, batch_size))
        while len(batch) > 0:
            for related_entity_lists in batch:
                print("\n {}".format(related_entity_lists))
            text = [
                vocab.tokenize_ids(related_entity_lists, remove_oov=False)
                for related_entity_lists in batch
            ]
            print("\n {}".format(text))
            pairs = text_to_pairs(text, vocab.random_ids,
                nsamples_per_word=nsamples,
                half_window_size=window)
            #print(pairs.shape)
            yield pairs
            batch = list(islice(documents, batch_size))
            #print(batch)
