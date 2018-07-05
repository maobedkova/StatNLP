import copy
import numpy as np
import nltk
from collections import Counter, defaultdict
from nltk.tag import str2tuple
from viterbi_smooth import ProbsCounts, EM, smooth_LM


def read_txt(filename):
    """Reading input file"""
    with open(filename, encoding="iso8859_2") as f:
        text = f.read()
        tokens = text.split()
        print("Tokens count", len(tokens))
        return [str2tuple(token)[0] for token in tokens]


def split_data(tokens):
    """Splitting data into test, heldout and train"""
    S = tokens[-40000:]         # test
    H = tokens[-60000:-40000]   # heldout
    T = tokens[:-60000]         # train
    return S, H, T


if __name__ == "__main__":
    cz_text = "textcz2.ptg.txt"
    en_text = "texten2.ptg.txt"

    tags = read_txt(en_text)
    S, H, T = split_data(tags)

    # Smoothing
    pc = ProbsCounts(T)
    lambdas = EM(H, pc)
    S_smoothed = smooth_LM(lambdas, S, pc)
