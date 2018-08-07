import copy
import nltk
import numpy as np
from collections import Counter, defaultdict
from nltk import str2tuple


"""Trigram model smoothing"""


class TriProbsCounts:
    """Class for handling probabilities"""
    def __init__(self, tags):
        """Getting uniform, unigram, bigram, trigram probs"""
        # Get unigram counts and probs
        self.unigr_counts = Counter(tags)
        self.unigr_probs = defaultdict(float)
        unigr_N = sum(self.unigr_counts.values())
        for entry in self.unigr_counts:
            self.unigr_probs[entry] = float(self.unigr_counts[entry]) / unigr_N

        # Get bigram counts and probs
        self.bigr_counts = Counter(nltk.bigrams(tags))
        self.bigr_probs = defaultdict(float)
        for entry in self.bigr_counts:
            self.bigr_probs[entry] = float(self.bigr_counts[entry]) / self.unigr_counts[entry[0]]

        # Get trigram counts and probs
        self.trigr_counts = Counter(nltk.trigrams(tags))
        self.trigr_probs = defaultdict(float)
        for entry in self.trigr_counts:
            self.trigr_probs[entry] = float(self.trigr_counts[entry]) / self.bigr_counts[(entry[0], entry[1])]

        # Get uniform probability
        self.unif_prob = 1. / len(self.unigr_counts)

    def get_probs(self, hist2, hist1, word):
        """Getting probabilities for a word"""
        # Get probability for a word. If not present in the data, prob == 0.0
        p1 = self.unigr_probs[word]
        p2 = self.bigr_probs[(hist1, word)]
        p3 = self.trigr_probs[(hist2, hist1, word)]
        # Assign uniform prob when history for a word is unknown
        if p2 == 0.:
            p2 = 1. / len(self.unigr_probs) if self.unigr_probs[hist1] == 0. else 0.
        if p3 == 0.:
            p3 = 1. / len(self.unigr_probs) if self.bigr_probs[(hist2, hist1)] == 0. else 0.
        return self.unif_prob, p1, p2, p3

    def EM(self, H):
        """Smoothing EM algorithm: obtain lambdas"""
        # Initialize probability dictionaries
        self.lambdas = [0.25, 0.25, 0.25, 0.25]    # initial lambdas
        expected_lambdas = [0., 0., 0., 0.]
        old_lambdas = [0., 0., 0., 0.]
        # While changes of lambdas are significant
        while all(el > 0.00001 for el in np.absolute(np.subtract(old_lambdas, self.lambdas))):
            old_lambdas = copy.deepcopy(self.lambdas)
            # Create histories
            hist1 = "###"
            hist2 = "###"
            for word in H:
                p0, p1, p2, p3 = self.get_probs(hist2, hist1, word)
                # Compute smoothed prob
                p_lambda = self.lambdas[0] * p0 + self.lambdas[1] * p1 + self.lambdas[2] * p2 + self.lambdas[3] * p3
                # Update histories
                hist2 = hist1
                hist1 = word
                # Compute expected counts of lambdas
                expected_lambdas[0] += (self.lambdas[0] * p0 / p_lambda)
                expected_lambdas[1] += (self.lambdas[1] * p1 / p_lambda)
                expected_lambdas[2] += (self.lambdas[2] * p2 / p_lambda)
                expected_lambdas[3] += (self.lambdas[3] * p3 / p_lambda)
            # Recompute lambdas
            self.lambdas = [el / sum(expected_lambdas) for el in expected_lambdas]

    def trigr_smooth(self, data):
        """Smoothing of the whole language model using computed lambdas"""
        p_smoothed = defaultdict(float)
        # Create histories
        hist1 = "###"
        hist2 = "###"
        for word in data:
            p0, p1, p2, p3 = self.get_probs(hist2, hist1, word)
            # Compute smoothed prob
            p_lambda = self.lambdas[0] * p0 + self.lambdas[1] * p1 + self.lambdas[2] * p2 + self.lambdas[3] * p3
            # Rewrite probabilities
            p_smoothed[(hist2, hist1, word)] = p_lambda
            # Update histories
            hist2 = hist1
            hist1 = word
        return p_smoothed

    def trans_probs(self, hist2, hist1, word):
        """Getting a transition probability for Viterbi decoding"""
        p0, p1, p2, p3 = self.get_probs(hist2, hist1, word)
        p_lambda = self.lambdas[0] * p0 + self.lambdas[1] * p1 + self.lambdas[2] * p2 + self.lambdas[3] * p3
        return p_lambda


"""Lexical model and unigram model smoothing"""


class LexProbsCounts:
    """Class for handling probabilities"""
    def __init__(self, data):
        """Getting uniform, word given tag and tag probs"""
        words, tags = data[0], data[1]
        self.w_t_counts = Counter([(words[i], tags[i]) for i in range(len(words) - 1)])
        self.t_counts = Counter(tags)
        self.words, self.tags = list(set(words)), list(set(tags))
        self.a = 2 ** (-20)
        self.V = len(self.words) * len(self.tags)   # vocabulary size
        self.N = len(words)        # data size

    def get_probs(self, word, tag):
        """Getting probabilities for a word and a tag"""
        if (word, tag) in self.w_t_counts:
            return self.w_t_counts[(word, tag)] / self.t_counts[tag]
        return 1. / self.V

    def lex_smooth(self, data):
        """Smoothing of the whole language model"""
        p_smoothed = defaultdict(float)
        for token in data:
            word, tag = str2tuple(token)
            p_smoothed[(word, tag)] = self.get_probs(word, tag)
        return p_smoothed

    def emis_probs(self, word, tag):
        """Getting an emission probability for Viterbi decoding"""
        return self.get_probs(word, tag)

    def emis_probs_BW(self, word, tag):
        """Getting an emission probability for Viterbi decoding after Baum-Welch"""
        if (tag, word) in self.w_t_counts:
            return self.w_t_counts[(tag, word)]
        return 1. / self.V


class InitProbsCounts(LexProbsCounts):
    def get_probs(self, word, tag):
        """Getting probabilities for a tag"""
        return (self.t_counts[tag] + self.a) / (self.N + self.a * self.V)

    def init_probs(self, word, tag):
        """Getting an initial probability for Viterbi decoding"""
        return self.get_probs(word, tag)


class BiProbsCounts:
    """Class for handling probabilities"""
    def __init__(self, tags):
        """Getting uniform, unigram, bigram, trigram probs"""
        # Get unigram counts and probs
        self.unigr_counts = Counter(tags)
        self.unigr_probs = defaultdict(float)
        unigr_N = sum(self.unigr_counts.values())
        for entry in self.unigr_counts:
            self.unigr_probs[entry] = float(self.unigr_counts[entry]) / unigr_N

        # Get bigram counts and probs
        self.bigr_counts = Counter(nltk.bigrams(tags))
        self.bigr_probs = defaultdict(float)
        for entry in self.bigr_counts:
            self.bigr_probs[entry] = float(self.bigr_counts[entry]) / self.unigr_counts[entry[0]]

        # Get uniform probability
        self.unif_prob = 1. / len(self.unigr_counts)

    def get_probs(self, hist1, word):
        """Getting probabilities for a word"""
        # Get probability for a word. If not present in the data, prob == 0.0
        p1 = self.unigr_probs[word]
        p2 = self.bigr_probs[(hist1, word)]
        # Assign uniform prob when history for a word is unknown
        if p2 == 0.:
            p2 = 1. / len(self.unigr_probs) if self.unigr_probs[hist1] == 0. else 0.
        return self.unif_prob, p1, p2

    def EM(self, H):
        """Smoothing EM algorithm: obtain lambdas"""
        # Initialize probability dictionaries
        self.lambdas = [0.25, 0.25, 0.25]    # initial lambdas
        expected_lambdas = [0., 0., 0.]
        old_lambdas = [0., 0., 0.]
        # While changes of lambdas are significant
        while all(el > 0.00001 for el in np.absolute(np.subtract(old_lambdas, self.lambdas))):
            old_lambdas = copy.deepcopy(self.lambdas)
            # Create histories
            hist1 = "###"
            for word in H:
                p0, p1, p2 = self.get_probs(hist1, word)
                # Compute smoothed prob
                p_lambda = self.lambdas[0] * p0 + self.lambdas[1] * p1 + self.lambdas[2] * p2
                # Update histories
                hist1 = word
                # Compute expected counts of lambdas
                expected_lambdas[0] += (self.lambdas[0] * p0 / p_lambda)
                expected_lambdas[1] += (self.lambdas[1] * p1 / p_lambda)
                expected_lambdas[2] += (self.lambdas[2] * p2 / p_lambda)
            # Recompute lambdas
            self.lambdas = [el / sum(expected_lambdas) for el in expected_lambdas]

    def bigr_smooth(self, data):
        """Smoothing of the whole language model using computed lambdas"""
        p_smoothed = defaultdict(float)
        # Create histories
        hist1 = "###"
        for word in data:
            p0, p1, p2 = self.get_probs(hist1, word)
            # Compute smoothed prob
            p_lambda = self.lambdas[0] * p0 + self.lambdas[1] * p1 + self.lambdas[2] * p2
            # Rewrite probabilities
            p_smoothed[(hist1, word)] = p_lambda
            # Update histories
            hist1 = word
        return p_smoothed

    def trans_probs(self, hist1, word):
        """Getting a transition probability for Viterbi decoding"""
        p0, p1, p2 = self.get_probs(hist1, word)
        p_lambda = self.lambdas[0] * p0 + self.lambdas[1] * p1 + self.lambdas[2] * p2
        return p_lambda
