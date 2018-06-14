import nltk
from collections import Counter
import numpy as np
import operator
import math


def read_txt(filename):
    """Reading input file"""
    with open(filename, encoding="iso8859_2") as f:
        text = f.read()
        tokens = text.split()
        print("Tokens count", len(tokens))
        return tokens


def get_unigrams(tokens):
    """Creating unigram dictionary"""
    unigr_dict = Counter(tokens)
    return unigr_dict


def get_bigrams(tokens, distance=50):
    """Creating bigrams for different distances and creating bigram dictionary"""
    # bigrams = nltk.bigrams(tokens)
    bigrams = []
    for cur_tok_id in range(0, len(tokens) - 1):
        for dist_tok_id in range(1 if distance == 1 else 2, distance + 1):
            if len(tokens) > (cur_tok_id + dist_tok_id):
                bigrams.append((tokens[cur_tok_id], tokens[cur_tok_id + dist_tok_id]))
    bigr_dict = Counter(bigrams)
    return bigr_dict


def pmi_metric(p_x, p_y, p_xy):
    """Applying pmi metric"""
    return math.log(p_xy / (p_x * p_y), 2)


def pmi_count(text, distance):
    """Obtaining dictionaries and retrieveing couts for pmi metric for each bigram"""
    tokens = read_txt(text)
    unigr_dict = get_unigrams(tokens)
    bigr_dict = get_bigrams(tokens, distance)
    print("Unigram count", len(unigr_dict), "sum", sum(unigr_dict.values()))
    print("Bigram count", len(bigr_dict), "sum", sum(bigr_dict.values()))
    pmi_dict = {}
    for bigram in bigr_dict:
        unigram_1 = float(unigr_dict[bigram[0]])
        unigram_2 = float(unigr_dict[bigram[1]])
        bigram_value = float(bigr_dict[bigram])
        if unigram_1 >= 10. and unigram_2 >= 10.:   # treshold for disregarding unigrams
            pmi = pmi_metric(unigram_1 / sum(unigr_dict.values()),
                            unigram_2 / sum(unigr_dict.values()),
                            bigram_value / sum(bigr_dict.values()))
            pmi_dict[bigram] = pmi
    return sorted(pmi_dict.items(), key=operator.itemgetter(1), reverse=True)


def out(text, distance):
    """Printing out the results"""
    print("=" * 30)
    print("Text " + text)
    print("=" * 30)
    pmi_dict = pmi_count(text, distance)
    print("Observed collocations for " + text + " using bigram distance " + str(distance))
    for el in pmi_dict[:20]:
        print(" ".join(el[0]), el[1])


if __name__ == "__main__":
    cz_text = "TEXTCZ1.txt"
    en_text = "TEXTEN1.txt"

    # out(cz_text, 1)     # Czech using normal bigrams
    # out(en_text, 1)     # English using normal bigrams
    out(cz_text, 50)    # Czech using bigrams with distance 1-50
    # out(en_text, 50)    # English using bigrams with distance 1-50

