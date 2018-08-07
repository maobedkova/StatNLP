import argparse
import copy as cp
# import itertools as itr
import numpy as np
import pickle as pkl
from collections import defaultdict
from nltk.tag import str2tuple
from smoothing import TriProbsCounts, BiProbsCounts, LexProbsCounts, InitProbsCounts
from viterbi import evaluate


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
    sup_T = tokens[:10000]  # train for supervised part of BW
    unsup_T = tokens[10000:-60000]  # train for unsupervised part of BW
    return S, H, sup_T, unsup_T


def split_sentences(tokens):
    """Splitting data into sentences"""
    sentences = [[str2tuple(token) for token in sent.split("\n") if token != ""]
                 for sent in "\n".join(tokens).split("###/###")]
    return [sentence for sentence in sentences if sentence != []]


def get_obs_states(tokens):
    """Getting states (columns) and observations (rows) for trellis"""
    token_tuples = [str2tuple(token) for token in tokens]
    obs, states = zip(*token_tuples)
    if "cz" in args.text:   # limit tags because of memory error
        new_states = [state[:2] for state in states if state]
        states = new_states
    return obs, states


def forward_backward(T, E, I, obs_ids):
    """Forward-Backward algorithm for Baum-Welch"""
    N = len(obs_ids)
    (num_states, num_obs) = E.shape

    # Matrices initialization
    # forward probabilities (F) of being at state i at time j
    F = np.zeros((num_states, N + 1))
    # backward probabilities (B) of being at state i at time j
    B = np.zeros((num_states, N + 1))
    # probability matrix (P) of being at state i at time j
    P = np.zeros((num_states, N))

    normalize = lambda matrix_col: matrix_col / np.sum(matrix_col)

    # Forward pass
    F[:, 0] = I     # fill first column of forward probs with initial probs
    for obs_id in range(N):
        f_row = np.matrix(F[:, obs_id])
        F[:, obs_id + 1] = f_row * np.matrix(T) * np.matrix(np.diag(E[:, obs_ids[obs_id]]))
        F[:, obs_id + 1] = normalize(F[:, obs_id + 1])

    # Backward pass
    B[:, -1] = 1.0      # fill last column of backward probs with ones
    for obs_id in range(N, 0, -1):
        b_col = np.matrix(B[:, obs_id]).transpose()
        B[:, obs_id - 1] = (np.matrix(T) * np.matrix(np.diag(E[:, obs_ids[obs_id - 1]])) * b_col).transpose()
        B[:, obs_id - 1] = normalize(B[:, obs_id - 1])

    # Final probabilities
    P = np.array(F) * np.array(B)
    P = P / np.sum(P, 0)

    return P, F, B


def baum_welch(data, iter_stopped, treshold):
# def baum_welch(data, iter_stopped, treshold, emit_p):
    """Baum-Welch algorithm"""
    obs, states = get_obs_states(data)
    obs, states = np.array(obs), np.array(states)
    unique_obs = list(sorted(set(obs)))
    unique_states = list(sorted(set(states)))
    # unique_bi_states = list(set(itr.product(unique_states, repeat=2)))
    # we won`t use trigram model for BW (too time- and memory-consuming); I couldn`t even initialize Theta matrix
    num_obs = len(unique_obs)
    num_states = len(unique_states)
    # num_bi_states = len(unique_bi_states)

    # Mapping
    # we will use numpy matrices for speeding up computations; thus we map words to indices
    obs_map_dict = dict((word, i) for i, word in enumerate(unique_obs))         # dict of observation id mappings
    inv_obs_map_dict = dict((value, key) for key, value in obs_map_dict.items()) # invert dict to map from id to token
    map2id = lambda d, tokens: [d[token] for token in tokens]    # mapping function
    obs_ids = np.array(map2id(obs_map_dict, obs))            # indices for words for the whole text

    if iter_stopped == 0:
        # Matrices initialization
        # transition matrix (T) from state to state (num_states*num_states)
        # T = np.ones((num_bi_states, num_states))
        # for i in range(num_bi_states):
        #     for j in range(num_states):
        #         T[i, j] = tpc.trans_probs(unique_bi_states[i][0], unique_bi_states[i][1], states[j])
        T = np.ones((num_states, num_states))
        for i in range(num_states):
            for j in range(num_states):
                T[i, j] = bpc.trans_probs(unique_states[i], unique_states[j])

        # emission matrix (E) from observation to state (num_states*num_obs)
        E = np.ones((num_states, num_obs))
        for i in range(num_states):
            for j in range(num_obs):
                E[i, j] = lpc.emis_probs(inv_obs_map_dict[j], unique_states[i])
        iteration = 0
    else:
        with open("trans_" + str(iter_stopped - 1), "rb") as t:
            T = pkl.load(t)
        with open("emis_" + str(iter_stopped - 1), "rb") as e:
            E = pkl.load(e)
        iteration = iter_stopped - 1

    # initial probabilities matrix (I)
    I = np.ones(num_states)
    for i in range(num_states):
        I[i] = ipc.init_probs("", unique_states[i])

    # matrix of probabilities of being at state a at time j and b at time j+1
    # Theta = np.zeros((num_bi_states, num_states, len(obs)))
    Theta = np.zeros((num_states, num_states, len(obs)))

    print("Learning started.")
    converged = False
    while not converged:
        iteration += 1
        print("Iteration", iteration)
        old_T = cp.deepcopy(T)
        old_E = cp.deepcopy(E)
        # Expectation step
        P, F, B = forward_backward(T, E, I, obs_ids)
        print("Forward-Backward finished.")

        # transition probabilities at each time
        # for a_id in range(num_bi_states):
        for a_id in range(num_states):
            for b_id in range(num_states):
                for c_id in range(len(obs)):
                    Theta[a_id, b_id, c_id] = F[a_id, c_id] * B[b_id, c_id + 1] * \
                                              old_T[a_id, b_id] * old_E[b_id, obs_ids[c_id]]
        Theta = Theta / np.sum(Theta, (0, 1))
        print("Theta updated.")

        # Update transition matrix (T)
        # for a_id in range(num_bi_states):
        for a_id in range(num_states):
            for b_id in range(num_states):
                T[a_id, b_id] = np.sum(Theta[a_id, b_id, :]) / np.sum(P[a_id, :])
        T = T / np.sum(T, 1)
        pkl.dump(T, open("trans_" + str(iteration), "wb"))   # backup T
        print("T updated.")

        # Update emission matrix (E)
        for a_id in range(num_states):
            print(a_id)
            for b_id in range(num_obs):
                r_b_id = np.array(np.where(obs_ids == b_id)) + 1
                E[a_id, b_id] = np.sum(P[a_id, r_b_id]) / np.sum(P[a_id, 1:])
        E = np.nan_to_num(E)
        E = E / np.sum(E, 1).reshape(num_states, 1)
        E = np.nan_to_num(E)
        pkl.dump(E, open("emis_" + str(iteration), "wb"))   # backup E
        print("E updated.")

        # Check convergence
        T_diff = np.linalg.norm(old_T - T)
        E_diff = np.linalg.norm(old_E - E)
        print("T diff", T_diff)
        print("E diff", E_diff)
        if T_diff < treshold and E_diff < treshold:
            converged = True

    def transform2dict(matrix, iter1, iter2, iter1_arr, iter2_arr):
        """Transformation from matrix to dictionary"""
        probs = defaultdict(float)
        # for a_id in range(num_bi_states):
        for a_id in range(iter1):
            for b_id in range(iter2):
                # probs[(unique_bi_states[a_id][0], unique_bi_states[a_id][1], unique_states[b_id])]
                # = matrix[a_id, b_id]
                probs[(iter1_arr[a_id], iter2_arr[b_id])] = matrix[a_id, b_id]
        return probs

    trans_probs = transform2dict(T, num_states, num_states, unique_states, unique_states)
    emis_probs = transform2dict(E, num_states, num_obs, unique_states, unique_obs)

    return trans_probs, emis_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--text', help="Text name")
    parser.add_argument('--iter', help="On which iteration it has stopped", default=0, type=int)
    parser.add_argument('--treshold', help="Difference for convergence", default=0.001, type=int)
    args = parser.parse_args()

    tokens = read_txt(args.text)
    S, H, sup_T, unsup_T = split_data(tokens)

    # === SUPERVISED PART ===
    # Estimating parameters

    # Smoothing trigram model (linear interpolation)
    # tpc = TriProbsCounts(get_obs_states(sup_T)[1])
    # tpc.EM(get_obs_states(H)[1])
    bpc = BiProbsCounts(get_obs_states(sup_T)[1])
    bpc.EM(get_obs_states(H)[1])

    # Smoothing lexical model (add lambda)
    lpc = LexProbsCounts(get_obs_states(sup_T))     # used in BW as raw params

    # Smoothing unigram model (add lambda)
    ipc = InitProbsCounts(sup_T)    # used in BW as raw params

    # === UNSUPERVISED PART ===

    # Learning parameters
    trans_probs, emis_probs = baum_welch(unsup_T, args.iter, args.treshold)

    # Smoothing transition probabilities
    bpc.bigr_probs = trans_probs

    # Smoothing lexical probabilities
    lpc.w_t_counts = emis_probs

    # === DECODING PART ===

    S_sents = split_sentences(S)

    if "en" in args.text:
        states = set(str2tuple(token)[1] for token in tokens)  # all tags in the data
        evaluate(S_sents, states, ipc, lpc, bpc, alpha=2 ** (-70), n=20, n_path=30, mode="bigr")
        # alpha for pruning, n for pruning, n_path for backtracking
    else:
        states = set(str2tuple(token)[1] for token in tokens if len(token) > 10)  # all tags in the data
        states = set([state[:2] for state in states])
        evaluate(S_sents, states, ipc, lpc, bpc, alpha=2 ** (-100), n=5, n_path=5, mode="bigr")
        # alpha for pruning, n for pruning, n_path for backtracking
