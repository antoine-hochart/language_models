from abc import ABC, abstractmethod

import nltk
import numpy as np

from collections import defaultdict
from functools import reduce

######################################################################
# utility functions to avoid the use of lambda functions
# (not supported by pickle on Windows)

def zero():
    return 0

def zerodict():
    return defaultdict(zero)

######################################################################
# Ngrams language model

class LanguageModel(ABC):
    @abstractmethod
    def __init__(self, texts):
        ...
    
    @abstractmethod
    def proba(self, token, history):
        ...

    @abstractmethod
    def generate_token(self, history):
        ...

    def generate_text(self, seed, n_sent=1):
        np.random.seed()
        eos=['.', '!', '?']
        text = []
        for _ in range(n_sent):
            sentence = seed.copy()
            while sentence[-1] not in eos:
                token = self.generate_token(sentence)
                sentence.append(token)
            text.append(sentence)
        return text


class Ngram(LanguageModel):
    def __init__(self, text, order=2):
        self.order = order
        # placeholder for ngram scores
        # self.scores = defaultdict(lambda: defaultdict(lambda: 0))
        self.scores = defaultdict(zerodict)
        # count of occurences
        for sentence in text:
            for ngram in nltk.ngrams(sentence, self.order):
                h = ' '.join(ngram[:-1]).strip()
                w = ngram[-1]
                self.scores[h][w] += 1
        # probabilities of occurences
        for h, counts in self.scores.items():
            total = float(sum(counts.values()))
            for token, count in counts.items():
                self.scores[h][token] = count / total

    def generate_key(self, history):
        n = len(history) - self.order + 1
        return ' '.join(history[n:]).strip()

    def proba(self, token, history):
        h = self.generate_key(history)
        if h in self.scores:
            p = self.scores[h].get(token, 0)
        else:
            p = None
        return p

    def generate_token(self, history):
        token = None
        h = self.generate_key(history)
        if h in self.scores:
            token = np.random.choice(
                list(self.scores[h].keys()),
                p=list(self.scores[h].values())
                )
        return token


class SimpleInterpolNgrams(LanguageModel):
    def __init__(self, text, order=2, weight=0.75):
        self.order = order
        self.weight = weight
        self.models = []
        for d in range(order):
            self.models.append(Ngram(text, d+1))
        self.vocab = list(self.models[0].scores[''].keys())
    
    def proba(self, token, history):
        avg = lambda x, y: (1 - self.weight) * x + self.weight * y
        p = reduce(avg, [model.proba(token, history) for model in self.models
                         if model.proba(token, history) is not None])
        return p

    def generate_token(self, history):
        token = np.random.choice(
            self.vocab,
            p=[self.proba(token, history) for token in self.vocab]
            )
        return token