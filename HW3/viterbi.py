# import math
import argparse
import operator
from nltk.tag import str2tuple
from smoothing import TriProbsCounts, LexProbsCounts, InitProbsCounts


def read_txt(filename):
    """Reading input file"""
    with open(filename, encoding="iso8859_2") as f:
        text = f.read()
        tokens = text.split()
        return tokens


def split_data(tokens):
    """Splitting data into test, heldout and train"""
    S = tokens[-40000:]  # test
    H = tokens[-60000:-40000]  # heldout
    T = tokens[:-60000]  # train
    return S, H, T


def split_sentences(tokens):
    """Splitting data into sentences"""
    sentences = [[str2tuple(token) for token in sent.split("\n") if token != ""]
                 for sent in "\n".join(tokens).split("###/###")]
    return [sentence for sentence in sentences if sentence != []]


def get_tags(tokens):
    """Getting just tags for every line"""
    return [str2tuple(token)[1] for token in tokens]


def get_obs_states(tokens):
    """Getting states (columns) and observations (rows) for trellis"""
    token_tuples = [str2tuple(token) for token in tokens]
    obs, states = zip(*token_tuples)
    if "cz" in args.text:   # limit tags because of memory error
        new_states = [state[:2] for state in states if state]
        states = new_states
    return obs, states


def viterbi4trigr(sent, states, alpha, n, n_path):
    """Viterbi decoding for trigram distribution"""
    T = [{}]
    states = list(sorted(states))
    path = {}

    # Fill in first column of the trellis
    for state in states:
        prob = ipc.init_probs(sent[0][0], state) * lpc.emis_probs(sent[0][0], state)

        # prob = math.log(ipc.init_probs(sent[0][0], state)) + math.log(lpc.emis_probs(sent[0][0], state))
        # handling underflow (worked worse than simple normalization)

        T[0][("<<start>>", state)] = prob
        if prob > alpha:  # initial pruning
            path[("<<start>>", state)] = prob

    # Fill in the rest of the trellis
    # Iterate through stages (row)
    for stage_id in range(1, len(sent)):
        T.append({})
        tmp_path = {}
        # Iterate through states (column)
        for state in states:
            tmp_probs = dict(((hist2, hist1), T[stage_id - 1][(hist2, hist1)] * tpc.trans_probs(hist2, hist1, state) *
                              lpc.emis_probs(sent[stage_id][0], state))
                             for hist2, hist1 in sorted(T[stage_id - 1], key=T[stage_id - 1].get, reverse=True)[:n])
            # additional pruning based on n likeliest states

            # tmp_probs = dict(
            #     ((hist2, hist1), T[stage_id - 1][(hist2, hist1)] + math.log(tpc.trans_probs(hist2, hist1, state)))
            #     for hist2, hist1, in sorted(T[stage_id - 1], key=T[stage_id - 1].get, reverse=True)[:n])
            # handling underflow (worked worse than simple normalization)

            max_state = max(tmp_probs.items(), key=operator.itemgetter(1))
            if max_state[1] > alpha:  # pruning
                hist2 = max_state[0][0]
                hist1 = max_state[0][1]
                T[stage_id][(hist1, state)] = max_state[1]  # create new state

                # T[stage_id][(hist1, state)] = math.log(lpc.emis_probs(sent[stage_id][0], state)) + max_state[1]
                # handling underflow (worked worse than simple normalization)

                # Extend probable paths
                for p in path:
                    if p[-1] == hist1 and p[-2] == hist2:
                        tmp_path[p + (state,)] = max_state[1]
        # Update actual paths (delete not extended, choose 10 most probable of extended ones)
        path = {key: tmp_path[key] for key in sorted(tmp_path, key=tmp_path.get, reverse=True)[:n_path]}

        # Normalization for avoiding underflow
        total_sum = sum(T[stage_id].values())
        for el in T[stage_id]:
            T[stage_id][el] /= total_sum

    # Choose the likeliest path
    max_path = max(path.items(), key=operator.itemgetter(1))[0]
    return max_path[1:]


def viterbi4bigr(sent, states, ipc, lpc, tpc, alpha, n, n_path):
    """Viterbi decoding for bigram distribution"""
    T = [{}]
    states = list(sorted(states))
    path = {}

    # Fill in first column of the trellis
    for state in states:
        prob = ipc.init_probs(sent[0][0], state) * lpc.emis_probs_BW(sent[0][0], state)
        # prob = math.log(ipc.init_probs(sent[0][0], state)) + math.log(lpc.emis_probs(sent[0][0], state))
        # handling underflow (worked worse than simple normalization)

        T[0][state] = prob
        if prob > alpha:  # initial pruning
            path[(state,)] = prob

    # Fill in the rest of the trellis
    # Iterate through stages (row)
    for stage_id in range(1, len(sent)):
        T.append({})
        tmp_path = {}
        # Iterate through states (column)
        for state in states:
            tmp_probs = dict((hist1, T[stage_id - 1][hist1] * tpc.trans_probs(hist1, state) *
                              lpc.emis_probs_BW(sent[stage_id][0], state))
                             for hist1 in sorted(T[stage_id - 1], key=T[stage_id - 1].get, reverse=True)[:n])
            # additional pruning based on n likeliest states

            # tmp_probs = dict(
            #     (hist1, T[stage_id - 1][hist1] + math.log(tpc.trans_probs(hist1, state)))
            #     for hist1 in sorted(T[stage_id - 1], key=T[stage_id - 1].get, reverse=True)[:n])
            # handling underflow (worked worse than simple normalization)

            max_state = max(tmp_probs.items(), key=operator.itemgetter(1))
            if max_state[1] > alpha:  # pruning
                hist1 = max_state[0]
                T[stage_id][state] = max_state[1]  # create new state

                # T[stage_id][(hist1, state)] = math.log(lpc.emis_probs(sent[stage_id][0], state)) + max_state[1]
                # handling underflow (worked worse than simple normalization)

                # Extend probable paths
                for p in path:
                    if p[-1] == hist1:
                        tmp_path[p + (state,)] = max_state[1]
        # Update actual paths (delete not extended, choose 10 most probable of extended ones)
        path = {key: tmp_path[key] for key in sorted(tmp_path, key=tmp_path.get, reverse=True)[:n_path]}

        # Normalization for avoiding underflow
        total_sum = sum(T[stage_id].values())
        for el in T[stage_id]:
            T[stage_id][el] /= total_sum

    # Choose the likeliest path
    max_path = max(path.items(), key=operator.itemgetter(1))[0]
    return max_path


def evaluate(sents, states, ipc, lpc, tpc, alpha, n, n_path, lang, mode="trigr"):
    """Computing accuracy (correct / total)"""
    correct = 0
    total = 0
    for sent in sents:
        if mode == "trigr":
            pred = viterbi4trigr(sent, states, alpha, n, n_path)
        else:
            pred = viterbi4bigr(sent, states, ipc, lpc, tpc, alpha, n, n_path)
        # print(sent, pred)
        for i in range(0, len(pred)):
            if lang == "en":
                actual = sent[i][1]
            else:
                actual = sent[i][1][:2]
            if pred[i] == actual:
                correct += 1
            total += 1
        print("Accuracy:", float(correct) / total, correct, "out of", total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--text', help="Text name")
    args = parser.parse_args()

    tokens = read_txt(args.text)
    S, H, T = split_data(tokens)

    # Smoothing trigram model (linear interpolation)
    tpc = TriProbsCounts(get_tags(T))
    tpc.EM(get_tags(H))

    # Smoothing lexical model (add lambda)
    lpc = LexProbsCounts(get_obs_states(T + H))

    # Smoothing unigram model (add lambda)
    ipc = InitProbsCounts(T + H)

    S_sents = split_sentences(S)

    if "en" in args.text:
        states = set(str2tuple(token)[1] for token in tokens)  # all tags in the data
        evaluate(S_sents, states, ipc, lpc, tpc, alpha=2 ** (-70), n=20, n_path=30, lang="en")
        # alpha for pruning, n for pruning, n_path for backtracking
    else:
        states = set(str2tuple(token)[1] for token in tokens if len(token) > 10)  # all tags in the data
        states = set([state[:2] for state in states])
        evaluate(S_sents, states, ipc, lpc, tpc, alpha=2 ** (-100), n=5, n_path=5, lang="cz")
        # alpha for pruning, n for pruning, n_path for backtracking
