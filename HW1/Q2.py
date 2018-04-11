import numpy
import copy
import matplotlib.pyplot as plt


def unigram_count_dict(data):
    """Creates dictionary of unigram counts"""
    uni_counts = {}

    for word in data[:-2]:
        if word in uni_counts:
            uni_counts[word] += 1
        else:
            uni_counts[word] = 1

    return uni_counts


def unigram_prob_dict(data):
    """Creates dictionary of unigram probabilities"""
    uni_counts = unigram_count_dict(data)
    uni_probs = {}

    for word in uni_counts:
        uni_probs[word] = float(uni_counts[word]) / len(data)

    return uni_counts, uni_probs


def bigram_count_dict(data):
    """Creates dictionary of bigram counts"""
    bi_counts = {}

    for i in range(0, len(data) - 2):
        if i + 1 <= len(data) - 1:
            if data[i] in bi_counts:
                if data[i + 1] in bi_counts[data[i]]:
                    bi_counts[data[i]][data[i + 1]] += 1
                else:
                    bi_counts[data[i]][data[i + 1]] = 1
            else:
                bi_counts[data[i]] = {data[i + 1]: 1}

    return bi_counts


def bigram_prob_dict(data):
    """Creates dictionary of bigram probabilities"""
    global uni_counts, bi_counts
    bi_counts = bigram_count_dict(data)

    bi_probs = {}

    for word1 in bi_counts:
        bi_probs[word1] = {}
        for word2 in bi_counts[word1]:
            bi_probs[word1][word2] = float(bi_counts[word1][word2]) / uni_counts[word1]

    return bi_counts, bi_probs


def trigram_count_dict(data):
    """Creates dictionary of trigram counts"""
    tri_counts = {}

    for i in range(0, len(data) - 1):
        if i + 2 <= len(data) - 1:
            if data[i] in tri_counts:
                if data[i + 1] in tri_counts[data[i]]:
                    if data[i + 2] in tri_counts[data[i]][data[i + 1]]:
                        tri_counts[data[i]][data[i + 1]][data[i + 2]] += 1
                    else:
                        tri_counts[data[i]][data[i + 1]][data[i + 2]] = 1
                else:
                    tri_counts[data[i]][data[i + 1]] = {data[i + 2]: 1}
            else:
                tri_counts[data[i]] = {data[i + 1]: {data[i + 2]: 1}}

    return tri_counts


def trigram_prob_dict(data):
    """Creates dictionary of trigram probabilities"""
    global bi_counts
    tri_counts = trigram_count_dict(data)
    tri_probs = {}

    for word1 in tri_counts:
        tri_probs[word1] = {}
        for word2 in tri_counts[word1]:
            tri_probs[word1][word2] = {}
            for word3 in tri_counts[word1][word2]:
                tri_probs[word1][word2][word3] = float(tri_counts[word1][word2][word3]) / bi_counts[word1][word2]

    return tri_counts, tri_probs


def get_probs(word, hist1, hist2):
    """Getting probabilities for a word"""
    p1_var = 0
    p2_var = 0
    p3_var = 0

    if word in p1:
        p1_var = p1[word]
        if hist1 in p2:
            if word in p2[hist1]:
                p2_var = p2[hist1][word]
                if hist2 in p3:
                    if hist1 in p3[hist2]:
                        if word in p3[hist2][hist1]:
                            p3_var = p3[hist2][hist1][word]

    # if history for word, use uniform probability
    if hist1 not in bi_counts or word not in bi_counts[hist1]:
        if hist1 not in uni_counts:
            p2_var = float(1) / len(p1)
    if hist2 not in tri_counts or hist1 not in tri_counts[hist2] or word not in tri_counts[hist2][hist1]:
        if hist2 not in bi_counts or hist1 not in bi_counts[hist2]:
            p3_var = float(1) / len(p1)

    return p1_var, p2_var, p3_var


def EM(data):
    """EM algorithm"""
    a = [0.25, 0.25, 0.25, 0.25]
    expected_a = [0, 0, 0, 0]
    old_a = [0, 0, 0, 0]

    while all(el > 0.00001 for el in numpy.absolute(numpy.subtract(old_a, a))):
        old_a = copy.deepcopy(a)

        # create histories
        hist1 = "ujudeg"
        hist2 = "rtyujo"

        for word in data:
            p1_var, p2_var, p3_var = get_probs(word, hist1, hist2)

            # compute smoothed prob
            p_lambda = a[0] * p0 + a[1] * p1_var + a[2] * p2_var + a[3] * p3_var

            # update histories
            hist2 = hist1
            hist1 = word

            # compute expected counts of lambdas
            expected_a[0] += (a[0] * p0 / p_lambda)
            expected_a[1] += (a[1] * p1_var / p_lambda)
            expected_a[2] += (a[2] * p2_var / p_lambda)
            expected_a[3] += (a[3] * p3_var / p_lambda)

        # recompute lambdas
        a = [el / sum(expected_a) for el in expected_a]

    return a


def smoothed_LM(lambdas, data):
    """Smoothed language model with computed lambdas"""
    p_lambdas = {}

    # create histories
    hist1 = "ujudeg"
    hist2 = "rtyujo"

    for word in data:
        # get probs
        p1_var, p2_var, p3_var = get_probs(word, hist1, hist2)

        # compute smoothed prob
        p_lambda = lambdas[0] * p0 + lambdas[1] * p1_var + lambdas[2] * p2_var + lambdas[3] * p3_var

        # write new probabilities
        if hist2 not in p_lambdas:
            p_lambdas[hist2] = {}
        if hist1 not in p_lambdas[hist2]:
            p_lambdas[hist2][hist1] = {}
        p_lambdas[hist2][hist1][word] = p_lambda

        # update histories
        hist2 = hist1
        hist1 = word

    return p_lambdas


def cross_entropy(data, smoothed):
    """Computation of cross entropy"""
    H = 0   # cross entropy
    # create histories
    hist1 = "ujudeg"
    hist2 = "rtyujo"

    train_seen = 0

    # count cross entropy
    for word in data:
        if word in uni_counts:
            train_seen += 1
        H += numpy.log2(smoothed[hist2][hist1][word])
        hist2 = hist1
        hist1 = word

    print("Coverage graph:", float(train_seen) / len(data))

    print (H*(-1) / len(data))
    return H*(-1) / len(data)


def change_lambdas(lambdas, mode, set, data):
    """Tweaking lambdas"""
    w = open(str(mode) + '.txt', 'w')
    w.write("Tweaking lambda parameter;Cross entropy\n")

    hs = []
    for perc in set:
        if mode == 1:
            diff = (1 - lambdas[3]) * perc
        else:
            diff = (-1) * lambdas[3] * (1 - perc)
        new_lambdas = [el - diff * el / sum(lambdas[0:3]) for el in lambdas]
        new_lambdas[3] = lambdas[3] + diff

        smoothed = smoothed_LM(new_lambdas, data)
        H = cross_entropy(data, smoothed)
        hs.append(H)

        w.write(str(perc) + ';' + str(H) + '\n')

    return hs


if __name__ == "__main__":
    # read a file
    hs_all = []

    f_names = ["TEXTEN1.txt", "TEXTCZ1.txt"]
    for f_name in f_names:
        print(f_name)
        f = open(f_name, 'r')
        text = f.read()
        text = text.split("\n")

        # create test, training and heldout data
        test_data = text[-20000:]
        text_tmp = text[:-20000]
        heldout_data = text_tmp[-40000:]
        train_data = text_tmp[:-39998]  # added two words to avoid bigram and trigram problems

        # create a dict of word counts
        uni_counts, p1 = unigram_prob_dict(train_data)  # unigram probability
        bi_counts, p2 = bigram_prob_dict(train_data)  # bigram probability
        tri_counts, p3 = trigram_prob_dict(train_data)  # trigram probability
        p0 = float(1) / len(uni_counts)  # uniform probability

        # language model
        print("Heldout data lambdas")
        lambdas = EM(heldout_data)
        print(lambdas)
        print("Training data lambdas", EM(train_data))

        # obtain test smoothed model
        test_smoothed = smoothed_LM(lambdas, test_data)

        # cross entropy computation
        cross_entropy(test_data, test_smoothed)

        # change lambdas
        print ("Increase lambda_3, discount other lambdas")
        difference_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        hs = change_lambdas(lambdas, 1, difference_set, test_data)
        hs_all.append(hs)

        print ("Decrease lambda_3, boost other lambdas")
        value_set = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
        hs = change_lambdas(lambdas, 2, value_set, test_data)
        hs_all.append(hs)

    ax = plt.gca()
    ax.grid(True)
    plt.title("Percentage of the difference between the trigram smoothing parameter and 1.0")
    plt.ylabel("Cross entropy")
    plt.xlabel("Tweaking lambda parameter (percentage)")
    plt.plot(difference_set, hs_all[0], label="en")
    plt.plot(difference_set, hs_all[2], label="cz")
    plt.legend(loc="best", frameon=False)
    plt.show()

    ax = plt.gca()
    ax.grid(True)
    plt.title("Percentage of the trigram smoothing parameter value")
    plt.ylabel("Cross entropy")
    plt.xlabel("Tweaking lambda parameter (percentage)")
    plt.plot(value_set, hs_all[1], label = "en")
    plt.plot(value_set, hs_all[3], label="cz")
    plt.legend(loc="best", frameon=False)
    plt.show()








