from best_friends import *
import math


def read_txt(filename, mode, limit):
    """Reading input file and dividing into two groups of tokens (limited and all)"""
    tokens = []
    with open(filename, encoding="iso8859_2") as f:
        for line in f:
            if mode == "w":
                tokens.append(line.strip().split("/")[0])
            else:
                tokens.append(line.strip().split("/")[1])
        return tokens[:limit]


def mi(c_x, c_y, c_xy, N):
    """Counting MI using formuala on slide 127"""
    return (c_xy / N) * math.log((c_xy * N) / (c_x * c_y), 2) if (c_xy * N) / (c_x * c_y) > 0 else 0


def mi_sum(unigr_dict, bigr_dict, mi_dict, N):
    """Summing MI for all bigrams in a text and writing each to a dictionary"""
    for bigram in bigr_dict:
        mi_dict[bigram] = mi(unigr_dict[bigram[0]], unigr_dict[bigram[1]], bigr_dict[bigram], N)
    return mi_dict


def calculate_sum(wf, bigr_dict, mi_dict):
    """Summation part, formula on slide 127"""
    sum_dict = {}
    for word in wf:
        sum_dict[word] = sum(mi_dict[bigram] for bigram in bigr_dict if bigram[0] == word or bigram[1] == word)
    return sum_dict


def check_key(key, dict):
    """Checking the presence in a dictionary"""
    return dict[key] if key in dict else 0


def calculate_sub(sum_dict, mi_dict, word_a, word_b):
    """Subtraction part, formula on slide 127"""
    return sum_dict[word_a] + sum_dict[word_b] - \
        check_key((word_a, word_b), mi_dict) - check_key((word_b, word_a), mi_dict)


def calculate_sum_lr(word_a, word_b, unigr_dict, bigr_dict, word_list, posit, N):
    """Calculate sum over left positioned and right positioned word a and word b"""
    sum_lr_ab = 0
    c_ab = unigr_dict[word_a] + unigr_dict[word_b]
    for word in word_list:
        c_lr = float(unigr_dict[word])
        if not (word == word_a or word == word_b):
            if posit == 0:
                c_lr_ab = bigr_dict[(word, word_a)] + bigr_dict[(word, word_b)]
            else:
                c_lr_ab = bigr_dict[(word_a, word)] + bigr_dict[(word_b, word)]
            sum_lr_ab += mi(c_lr, c_ab, c_lr_ab, N)
    return sum_lr_ab, c_ab


def calculate_add(word_a, word_b, unigr_dict, bigr_dict, word_list, N):
    """Calculate add, formula on slide 128"""
    sum_l_ab, c_l_ab = calculate_sum_lr(word_a, word_b, unigr_dict, bigr_dict, word_list[0], 0, N)
    sum_r_ab, c_r_ab = calculate_sum_lr(word_a, word_b, unigr_dict, bigr_dict, word_list[1], 1, N)
    c_ab_ab = bigr_dict[(word_a, word_a)] + bigr_dict[(word_a, word_b)] + \
              bigr_dict[(word_b, word_a)] + bigr_dict[(word_b, word_b)]
    return mi(c_l_ab, c_r_ab, c_ab_ab, N) + sum_l_ab + sum_r_ab


def loss_count(wf, bigr_dict, unigr_dict, mi_dict, word_list, N):
    """Losses calculation, formula from slide 131, finding minimal loss"""
    L = {}
    L_min = ("", 1000)
    sum_dict = calculate_sum(wf, bigr_dict, mi_dict)
    for id_a in range(0, len(wf) - 1):
        for id_b in range(id_a + 1, len(wf) - 1):   # so that not to repeat
            word_a = wf[id_a]
            word_b = wf[id_b]
            sub = calculate_sub(sum_dict, mi_dict, word_a, word_b)
            add = calculate_add(word_a, word_b, unigr_dict, bigr_dict, word_list, N)
            L[(word_a, word_b)] = sub - add
            if L_min[1] > L[(word_a, word_b)]:
                L_min = ((word_a, word_b), L[(word_a, word_b)])
    return L_min


def out(text, mode, word_limit, class_limit):
    """Printing out the results"""
    print("=" * 30)
    print("Text " + text)
    print("=" * 30)

    # Get tokens
    tokens = read_txt(text, mode, word_limit)
    N = float(len(tokens))
    print("Tokens", N)

    # Get unigram and bigram dictionaries
    unigr_dict = get_unigrams(tokens)
    bigr_dict = get_bigrams(tokens, 1)
    print("Unigram count", len(unigr_dict), "sum", sum(unigr_dict.values()))
    print("Bigram count", len(bigr_dict), "sum", sum(bigr_dict.values()))

    # Create class for every word
    classes = {}
    for word in unigr_dict:     # use limit 10 for building classes
        if unigr_dict[word] >= 10.:
            classes[word] = word

    # Set of left and right positioned words
    word_l, word_r = [], []
    for bigram in bigr_dict:
        word_l.append(bigram[0])
        word_r.append(bigram[1])
    word_l, word_r = set(word_l), set(word_r)


    mi_dict = {}
    while len(classes) != class_limit:
        mi_dict = mi_sum(unigr_dict, bigr_dict, mi_dict, N)

        L_min = loss_count(sorted(list(classes.keys())), bigr_dict, unigr_dict, mi_dict, [word_l, word_r], N)
        print(L_min)


        # count = 0
        # for key in mi_dict:
        #     if count == 10:
        #         break
        #     print(key, mi_dict[key])
        #     count += 1
        break



if __name__ == "__main__":
    cz_text = "TEXTCZ1.ptg"
    en_text = "TEXTEN1.ptg"

    out(en_text, "w", 8000, 15)       # English for words
