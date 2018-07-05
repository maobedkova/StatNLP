import numpy as np
import copy
from collections import Counter, defaultdict
import nltk


class ProbsCounts:
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

    def get_probs(self, word, hist1, hist2):
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


def EM(H, pc):
    """Smoothing EM algorithm: obtain lambdas"""
    # Initialize probability dictionaries
    lambdas = [0.25, 0.25, 0.25, 0.25]    # initial lambdas
    expected_lambdas = [0., 0., 0., 0.]
    old_lambdas = [0., 0., 0., 0.]
    # While changes of lambdas are significant
    while all(el > 0.00001 for el in np.absolute(np.subtract(old_lambdas, lambdas))):
        old_lambdas = copy.deepcopy(lambdas)
        # Create histories
        hist1 = "ujudeg"
        hist2 = "rtyujo"
        for word in H:
            p0, p1, p2, p3 = pc.get_probs(word, hist1, hist2)
            # Compute smoothed prob
            p_lambda = lambdas[0] * p0 + lambdas[1] * p1 + lambdas[2] * p2 + lambdas[3] * p3
            # Update histories
            hist2 = hist1
            hist1 = word
            # Compute expected counts of lambdas
            expected_lambdas[0] += (lambdas[0] * p0 / p_lambda)
            expected_lambdas[1] += (lambdas[1] * p1 / p_lambda)
            expected_lambdas[2] += (lambdas[2] * p2 / p_lambda)
            expected_lambdas[3] += (lambdas[3] * p3 / p_lambda)
        # Recompute lambdas
        lambdas = [el / sum(expected_lambdas) for el in expected_lambdas]
    return lambdas


def smooth_LM(lambdas, data, pc):
    """Smoothing of language model using computed lambdas"""
    p_smoothed = defaultdict(float)
    # Create histories
    hist1 = "ujudeg"
    hist2 = "rtyujo"
    for word in data:
        p0, p1, p2, p3 = pc.get_probs(word, hist1, hist2)
        # Compute smoothed prob
        p_lambda = lambdas[0] * p0 + lambdas[1] * p1 + lambdas[2] * p2 + lambdas[3] * p3
        # Rewrite probabilities
        p_smoothed[(hist2, hist1, word)] = p_lambda
        # Update histories
        hist2 = hist1
        hist1 = word
    return p_smoothed
