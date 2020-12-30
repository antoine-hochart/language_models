import os
import re
import codecs
import nltk
import numpy as np

from collections import Counter

######################################################################

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk.data.path[1])

######################################################################

proust = {
    'fname': [
        os.path.join('proust', 'recherche_{:02d}.txt'.format(i))
        for i in range(1,11)
        ],
    'start': [81, 42, 42, 41, 114, 91, 91, 74, 84, 91],
    'end': [16211, 6452, 6125, 6812, 6617, 7650, 7804, 9352, 9150, 7804], 
    }

######################################################################

def get_tokenized_text(fname, start=None, end=None):
    """
    Read a text from a file, divide it in sentences and tokenize them.
    Returns a list of lists:
    the outer list contains sentences, the inner lists contain words.
    """
    fpath = os.path.join(os.path.dirname(__file__), '..', '..', 'data', fname)
    with codecs.open(fpath, 'r', encoding='utf-8') as f:
        text = f.readlines()
    text = [line.strip().lower() for line in text[start:end]]
    text = nltk.sent_tokenize(' '.join(text))
    text = [sentence.replace("\'", " \' ") for sentence in text]
    # tokenizer = nltk.RegexpTokenizer(r'\w+')
    # text = [tokenizer.tokenize(sentence) for sentence in text]
    text = [nltk.word_tokenize(sentence) for sentence in text]
    return text


def get_corpus(text_info=proust):
    text = []
    for params in [dict(zip(text_info.keys(), values))
                   for values in zip(*text_info.values())]:
        text += get_tokenized_text(**params)
    n_sentences = len(text)
    n_words = sum(len(sentences) for sentences in text)
    print("Corpus with {} sentences and {} words".format(n_sentences, n_words))
    return text


def split_text(text, val_size=0.1, test_size=0.1, seed=0):
    """
    Split text into train/validation/test set.
    """
    # set indices of val and test sets
    n_sentences = len(text)
    val_idx = int((1 - val_size - test_size) * n_sentences)
    test_idx = int((1 - test_size) * n_sentences)
    # shuffle sentence indices
    np.random.seed(seed)
    indices = np.random.permutation(n_sentences)
    # set indices of train, val and test sets
    train_indices = set(indices[:val_idx])
    val_indices = set(indices[val_idx:test_idx])
    test_indices = set(indices[test_idx:])
    # create train/val/test sets
    train_text = [text[i] for i in train_indices]
    val_text = [text[i] for i in val_indices]
    test_text = [text[i] for i in test_indices]
    return (train_text, val_text, test_text)


def set_vocabulary(text, min_count=1, max_vocab_size=10**5):
    vocab = Counter([token for sentence in text for token in sentence])
    vocab = vocab.most_common(max_vocab_size)
    # compute <oov> token count
    n_oov = sum(count for (token, count) in vocab if count < min_count)
    # reduce vocabulary
    vocab = [(token, count) for (token, count) in vocab if count >= min_count]
    # add <oov> tokens
    vocab = [('<oov>', n_oov)] + vocab
    vocab = sorted(vocab, key=lambda x: x[1], reverse=True)
    print("Vocabulary size: {}".format(len(vocab)))
    return [token for token, count in vocab]


def replace_words(texts, vocab):
    """
    Replace out-of-vocabulary words with <oov> token.
    """
    texts = [[[token if token in vocab else '<oov>' for token in sentence]
                for sentence in text] for text in texts]
    return texts


def preprocess_text(text, val_size=0.1, test_size=0.1, min_count=1,
                    max_vocab_size=10**5, seed=0):
    texts = split_text(text, val_size, test_size, seed)
    vocab = set_vocabulary(texts[0], min_count, max_vocab_size)
    texts = replace_words(texts, set(vocab))
    return texts, vocab


######################################################################

def list2str(text):
    output = []
    for sentence in text:
        sentence = ' '.join(sentence)
        sentence = sentence.replace(" \' ", "\'")
        sentence = re.sub(r'\s([,;:!?.](?:\s|$))', r'\1', sentence)
        output.append(sentence)
    return '\n\n'.join(output)