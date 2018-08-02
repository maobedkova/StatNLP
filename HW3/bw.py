import copy as cp
import numpy as np
import pickle as pkl
import argparse
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
    sup_T = tokens[:10000]  # train for supervised part of BW
    unsup_T = tokens[10000:-60000]  # train for unsupervised part of BW
    return S, H, sup_T, unsup_T


def get_obs_states(tokens):
    """Getting states (columns) and observations (rows) for trellis"""
    token_tuples = [str2tuple(token) for token in tokens]
    obs, states = zip(*token_tuples)
    return obs, states


def forward_backward(T, E, I, obs):
    """Forward-Backward algorithm for Baum-Welch"""
    N = len(obs)
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
    for obs_id in range(num_obs):
        f_row = np.matrix(F[:, obs_id])
        F[:, obs_id + 1] = f_row * np.matrix(T) * np.matrix(np.diag(E[:, obs[obs_id]]))
        F[:, obs_id + 1] = normalize(F[:, obs_id + 1])

    # Backward pass
    B[:, -1] = 1.0      # fill last column of backward probs with ones
    for obs_id in range(num_obs, 0, -1):
        b_col = np.matrix(B[:, obs_id]).transpose()
        B[:, obs_id - 1] = (np.matrix(T) * np.matrix(np.diag(E[:, obs[obs_id - 1]])) * b_col).transpose()
        B[:, obs_id - 1] = normalize(B[:, obs_id - 1])

    # Final probabilities
    P = np.array(F) * np.array(B)
    P = P / np.sum(P, 0)

    return F, B, P


def baum_welch(data):
    """Baum-Welch algorithm"""
    obs, states = get_obs_states(data)
    obs, states = np.array(obs), np.array(states)

    num_obs = len(set(obs))
    num_states = len(set(states))

    # Mapping
    # we will use numpy matrices for speeding up computations; thus we map words to indices
    obs_map_dict = dict((word, i) for i, word in enumerate(set(obs)))           # dict of observation id mappings
    states_map_dict = dict((word, i) for i, word in enumerate(set(states)))     # dict of state id mappings
    map2id = lambda d, tokens: [d[token] for token in tokens]    # mapping function
    obs_ids = map2id(obs_map_dict, obs)             # indices for words for the whole text
    states_ids = map2id(states_map_dict, states)    # indices for tags for the whole text

    # Matrices initialization
    # transition matrix (T) from state to state (num_states*num_states)
    T = np.ones((num_states, num_states))
    T = T / num_states    # uniform probs init bcz can`t fill with what we have
    # emission matrix (E) from observation to state (num_states*num_obs)
    E = np.ones((num_states, num_obs))
    for i in range(num_states):
        for j in range(num_obs):
            E[i, j] = lpc.emis_probs(obs[j], states[i])
    # initial probabilities matrix (I)
    init_iterable = (ipc.init_probs("", states[i]) for i in range(num_states))
    I = np.fromiter(init_iterable, float)
    # matrix of probabilities of being at state a at time j and b at time j+1
    Theta = np.zeros((num_states, num_states, len(obs)))

    print("Learning started.")
    converged = False
    iteration = 0
    while not converged:
        iteration += 1
        print("Iteration", iteration)
        old_T = cp.deepcopy(T)
        old_E = cp.deepcopy(E)
        # Expectation step
        P, F, B = forward_backward(old_T, old_E, I, obs_ids)
        print("Forward-Backward finished.")
        T = np.ones((num_states, num_states))
        E = np.ones((num_states, num_obs))

        # transition probabilities at each time
        for a_id in range(num_states):
            print(a_id)
            for b_id in range(num_states):
                for c_id in range(len(obs)):
                    Theta[a_id, b_id, c_id] = F[a_id, c_id] * B[b_id, c_id + 1] * \
                                              old_T[a_id, b_id] * old_E[b_id, obs_ids[c_id]]

        # Update transition matrix (T)
        for a_id in range(num_states):
            for b_id in range(num_states):
                T[a_id, b_id] = np.sum(Theta[a_id, b_id, :]) / np.sum(P[a_id, :])
        T = T / np.sum(T, 1)
        pkl.dump(T, open("trans_" + iteration, "wb"))   # backup T

        # Update emission matrix (E)
        for a_id in range(num_states):
            for b_id in range(num_obs):
                r_b_id = np.array(np.where(obs == b_id)) + 1
                E[a_id, b_id] = np.sum(P[a_id, r_b_id]) / np.sum(P[a_id, 1:])
        E = E / np.sum(E, 1)
        pkl.dump(E, open("emis_" + iteration, "wb"))   # backup E

        # Check convergence
        if np.linalg.norm(old_T - T) < .001 and np.linalg.norm(old_E - E) < .001:
            converged = True

    return T, E


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--text', help="Text name")
    args = parser.parse_args()

    tokens = read_txt(args.text)
    S, H, sup_T, unsup_T = split_data(tokens)

    # === SUPERVISED PART ===
    # Estimating parameters

    # Smoothing trigram model (linear interpolation)
    tpc = TriProbsCounts(get_obs_states(sup_T)[1])
    tpc.EM(get_obs_states(H)[1])

    # Smoothing lexical model (add lambda)
    lpc = LexProbsCounts(sup_T)

    # Smoothing unigram model (add lambda)
    ipc = InitProbsCounts(sup_T)

    # === UNSUPERVISED PART ===

    baum_welch(unsup_T)
