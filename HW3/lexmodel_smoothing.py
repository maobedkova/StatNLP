from collections import Counter, defaultdict
from nltk import str2tuple


class ProbsCounts:
    """Class for handling probabilities"""
    def __init__(self, tokens):
        """Getting uniform, word given tag and tag probs"""
        self.w_t_counts = Counter([str2tuple(token) for token in tokens])
        self.t_counts = Counter([str2tuple(token)[1] for token in tokens])
        self.words, self.tags = zip(*list(self.w_t_counts.keys()))
        self.unif_prob = 1. / (len(self.words) * len(self.tags))

    def get_probs(self, word, tag):
        """Getting probabilities for a word and a tag"""
        if (word, tag) in self.w_t_counts:
            return self.w_t_counts[(word, tag)] / self.t_counts[tag]
        return self.unif_prob


def smooth_LM(data, pc):
    """Smoothing of language model"""
    p_smoothed = defaultdict(float)
    for token in data:
        word, tag = str2tuple(token)
        p_smoothed[(word, tag)] = pc.get_probs(word, tag)
    return p_smoothed
