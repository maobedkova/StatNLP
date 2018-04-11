import copy
import math
import numpy
import random
import re
import matplotlib.pyplot as plt


def create_dict(text):
    """Creating a dict to measure probability"""
    bigrams = {}    # dictionary of bigrams
    sum_count = 0   # overall number of bigrams

    # creating a dict of bigrams
    for i in range(0, len(text) - 1):
        if text[i + 1]:
            if text[i] in bigrams:
                if text[i + 1] in bigrams[text[i]]:
                    bigrams[text[i]][text[i + 1]] += 1
                else:
                    bigrams[text[i]][text[i + 1]] = 1
            else:
                bigrams[text[i]] = {text[i + 1]: 1}
            sum_count += 1

    return bigrams, sum_count


def gather_info(text):
    """Gathering all words and characters from a text"""
    global alphabet, dictionary

    word_count = 0  # count of words in a text
    letter_count = 0    # count of letters in a text
    dict = {}   # dictionary for frequencies
    for word in text:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
            letter_count += 1
        if word not in dictionary:
            dictionary.append(word)
        word_count += 1

    unique_word = 0 # count of words with frequency = 1
    for word in dict:
        if dict[word] == 1:
            unique_word += 1

    print("Word count:", word_count)
    print("Word form count:", len(dict))
    print("Letter count:", letter_count)
    print("Letter count per word:", letter_count / word_count)
    print("Unique word count:", unique_word)


def count_H_Px(bigrams, sum_count):
    """Computation of conditional entropy and perplexity"""
    unique_bigr = 0
    bigr_count = 0
    H = 0   # conditional entropy
    for word1 in bigrams:
        sum_word1 = sum(bigrams[word1].values())
        for word2 in bigrams[word1]:
            if bigrams[word1][word2] == 1:
                unique_bigr += 1
            bigr_count += 1
            bigrams[word1][word2] = (float(bigrams[word1][word2]) / sum_count,
                                     float(bigrams[word1][word2]) / sum_word1)
            H += bigrams[word1][word2][0] * math.log(bigrams[word1][word2][1], 2)

    # Px = math.pow(2, H*(-1))     # perplexity
    Px = numpy.power(2, H * (-1))   # perplexity

    print ("Bigram count:", bigr_count)
    print ("Unique bigram count:", unique_bigr)

    print ("Conditional entropy:", H*(-1))
    print ("Perplexity:", Px)

    return H*(-1), Px


def messing_up(text, n, arr, mark):
    """Messing up characters or words in a text"""
    # conditional entropy numbers for graphs
    hs = []

    # write down entropy
    w = open("data_written.csv", 'w')
    w.write('Likelihood;Minimum;Maximum;Average\n')

    # likelihoods of messing up
    for x in [0.1, 0.05, 0.01, 0.001, 0.0001]:
        print ("Likelihod =", x)
        w.write(str(x) + ";")
        hs_tmp = []
        for run in range(1, 11):
            print ("run", run)
            sample = random.sample(range(0, n - 1), math.ceil(n * x))
            new_text = copy.deepcopy(text)  # new copy of a text
            for i in sample:
                # condition for messing up characters
                if text[i] == '\n':
                    r = random.randint(0, len(text) - 1)
                    while r in sample or text[r] == '\n':
                        r = numpy.random.randint(0, len(text) - 1)
                    new_text[r] = numpy.random.choice(arr)
                else:
                    new_text[i] = numpy.random.choice(arr)
            # condition for messing up characters
            if mark == 'c':
                new_text = "".join(new_text).split('\n')
            # count conditional entropy and perplexity of a messed up text
            bigrams_inner, sum_count_inner = create_dict(new_text)
            H, Px = count_H_Px(bigrams_inner, sum_count_inner)
            hs_tmp.append(H)
        hs.append(numpy.mean(hs_tmp))

        w.write(str(min(hs_tmp)) + ";" + str(max(hs_tmp)) + ";" + str(numpy.mean(hs_tmp)) + "\n")
    return hs


def work_with_files(file_name):
    # read a file
    f = open(file_name, 'r')
    text = f.read()

    # retrieve all words and characters
    gather_info(text.split('\n'))

    # compute conditional entropy and perplexity
    bigrams, sum_count = create_dict(text.split('\n'))
    count_H_Px(bigrams, sum_count)

    # mess up characters, then words
    print ("Messing up characters...")
    hs_c = messing_up(list(text), len(list(re.sub('\n', '', text))), alphabet, "c")
    print("=" * 30)
    print ("Messing up words...")
    hs_w = messing_up(text.split('\n'), len(text.split('\n')), dictionary, "w")

    return hs_c, hs_w


if __name__ == "__main__":
    # working with English text
    print("English text")
    alphabet = []
    dictionary = []
    en_hs_c, en_hs_w = work_with_files("TEXTEN1.txt")
    print("=" * 30)

    # plotting for English
    ax = plt.gca()
    ax.grid(True)
    plt.title("English text. Messed up characters vs. messed up words")
    plt.ylabel("Conditional entropy")
    plt.xlabel("Likelihood")
    plt.plot([0.1, 0.05, 0.01, 0.001, 0.0001], en_hs_c, label="characters")
    plt.plot([0.1, 0.05, 0.01, 0.001, 0.0001], en_hs_w, label="words")
    plt.legend(loc="best", frameon=False)
    plt.show()

    # working with Czech text
    print("Czech text")
    alphabet = []
    dictionary = []
    cz_hs_c, cz_hs_w = work_with_files("TEXTCZ1.txt")
    print("=" * 30)

    # plotting for Czech
    ax = plt.gca()
    ax.grid(True)
    plt.title("Czech text. Messed up characters vs. messed up words")
    plt.ylabel("Conditional entropy")
    plt.xlabel("Likelihood")
    plt.plot([0.1, 0.05, 0.01, 0.001, 0.0001], cz_hs_c, label="characters")
    plt.plot([0.1, 0.05, 0.01, 0.001, 0.0001], cz_hs_w, label="words")
    plt.legend(loc="best", frameon=False)
    plt.show()


    print("English vs. Czech plotting...")
    # plotting for characters
    ax = plt.gca()
    ax.grid(True)
    plt.title("Messed up characters. English text vs. Czech text")
    plt.ylabel("Conditional entropy")
    plt.xlabel("Likelihood")
    plt.plot([0.1, 0.05, 0.01, 0.001, 0.0001], en_hs_c, label="en")
    plt.plot([0.1, 0.05, 0.01, 0.001, 0.0001], cz_hs_c, label="cz")
    plt.legend(loc="best", frameon=False)
    plt.show()

    # plotting for words
    ax = plt.gca()
    ax.grid(True)
    plt.title("Messed up words. English text vs. Czech text")
    plt.ylabel("Conditional entropy")
    plt.xlabel("Likelihood")
    plt.plot([0.1, 0.05, 0.01, 0.001, 0.0001], en_hs_w, label="en")
    plt.plot([0.1, 0.05, 0.01, 0.001, 0.0001], cz_hs_w, label="cz")
    plt.legend(loc="best", frameon=False)
    plt.show()
