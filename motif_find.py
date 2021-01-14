# Lior Paz 206240996, Shimmy Balsam 204693352

import argparse
import numpy as np
from scipy.special import logsumexp

ALPHABET_LEN = 6

EMPTY_STRING = ""

NOT_MOTIF_STATES = 4

STATES_AT_END = 3

FIRST_MOTIF_STATE = 2

BACKGROUND = 'B'

MOTIF = 'M'

LINE_LENGTH = 50

B_END = 'B_End'

B_2 = 'B_2'

B_1 = 'B_1'

B_START = 'B_Start'

START_SIGN = '^'
END_SIGN = '$'
END_STATE_EMISSIONS = {"A": 0, "C": 0, "G": 0, "T": 0, START_SIGN: 0, END_SIGN: 1}

START_STATE_EMISSIONS = {"A": 0, "C": 0, "G": 0, "T": 0, START_SIGN: 1, END_SIGN: 0}

UNIFORM_PROB = 0.25

UNIFORM_DIST_DICT = {"A": UNIFORM_PROB, "C": UNIFORM_PROB, "G": UNIFORM_PROB, "T": UNIFORM_PROB, START_SIGN: 0,
                     END_SIGN: 0}

alphabet = ['A', 'C', 'G', 'T']

emission_dict = {START_SIGN: 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, END_SIGN: 5}


def print_results(seq, S):
    """
    prints by the required format
    :param seq: original sequence
    :param S: motifs assigment
    """
    length = len(seq)
    for i in range(0, length, LINE_LENGTH):
        cur_min = min(i + LINE_LENGTH, length)
        print(S[i:cur_min])
        print(seq[i:cur_min] + '\n')


def parse_emission_matrix(emissions_matrix_path):
    """
    takes the emission matrix file and parse it into a numpy matrix
    :param emissions_matrix_path: path to file
    :return: emission table, number of states
    """
    f = open(emissions_matrix_path)
    f.readline()  # remove first line
    lines = f.readlines()
    k_counter = len(lines)
    emissions_mat = np.zeros([k_counter + NOT_MOTIF_STATES, ALPHABET_LEN])
    # B start
    emissions_mat[0, 0] = 1
    # B end
    emissions_mat[-1, -1] = 1
    # B_1
    emissions_mat[1, 1:-1] = UNIFORM_PROB
    # B_2
    emissions_mat[-2, 1:-1] = UNIFORM_PROB
    for k, line in enumerate(lines, 2):  # go over every line
        emissions = line.split('	')
        for letter in range(len(alphabet)):  # create emissions for every S_i
            emissions_mat[k, letter + 1] = float(emissions[letter])
    return wrap_log(emissions_mat), k_counter


def build_transition_matrix(k_counter, q, p):
    """
    build transition matrix according to number of states and given q,p
    :param k_counter: num of states
    :param q: probability of motif in seq
    :param p: probability to enter motif
    :return: transition matrix
    """
    dim = k_counter + NOT_MOTIF_STATES
    transition_mat = np.zeros([dim, dim])
    # B start
    transition_mat[0, 1] = q
    transition_mat[0, -2] = 1 - q
    # B_1
    transition_mat[1, 1:3] = [1 - p, p]
    # B_2
    transition_mat[-2, -2:] = [1 - p, p]
    # S_last
    transition_mat[-3, -2] = 1
    # all S
    transition_mat[2:-3, 3:-2] = np.eye(k_counter - 1)
    return wrap_log(transition_mat)


def forward(seq, emission_mat, transition_mat, k_counter):
    """
    calculate the forward table for a given seq
    :param seq: sequence
    :param emission_mat
    :param transition_mat
    :param k_counter: number of states
    :return: Forward table
    """
    k_dim = k_counter + NOT_MOTIF_STATES
    N = len(seq)
    forward_table = wrap_log(np.zeros([k_dim, N]))
    forward_table[0, 0] = wrap_log(1)
    for j in range(1, N):
        curr_letter = forward_table[:, j - 1].reshape(-1, 1)
        forward_table[:, j] = logsumexp(curr_letter + transition_mat, axis=0) + emission_mat[:, emission_dict[seq[j]]]
    return forward_table


def backward(seq, emission_mat, transition_mat, k_counter):
    """
    calculate the backward table for a given seq
    :param seq: sequence
    :param emission_mat
    :param transition_mat
    :param k_counter: number of states
    :return: Backward table
    """
    k_dim = k_counter + NOT_MOTIF_STATES
    N = len(seq)
    backward_table = wrap_log(np.zeros([k_dim, N]))
    backward_table[-1, -1] = wrap_log(1)
    for j in range(N - 2, -1, -1):
        curr_letter = backward_table[:, j + 1].reshape(-1, 1)
        backward_table[:, j] = logsumexp(
            curr_letter + transition_mat.T + emission_mat[:, emission_dict[seq[j + 1]]].reshape((-1, 1)), axis=0)
    return backward_table


def posterior(seq, emission_mat, transition_mat, k_counter):
    """
    calculates the most probable state for every base in seq
    :param seq: sequence
    :param emission_mat
    :param transition_mat
    :param k_counter: num of states
    :return: seq of states, aligned to original seq
    """
    k_dim = k_counter + NOT_MOTIF_STATES
    N = len(seq)
    forward_table = forward(seq, emission_mat, transition_mat, k_counter)
    backward_table = backward(seq, emission_mat, transition_mat, k_counter)
    posterior_table = forward_table + backward_table
    motif_order = EMPTY_STRING
    # decide states
    for j in range(N):
        curr_k = int(np.argmax(posterior_table[:, j]))
        last_motif_state = k_dim - STATES_AT_END
        if FIRST_MOTIF_STATE <= curr_k <= last_motif_state:
            motif_order += MOTIF
        else:
            motif_order += BACKGROUND
    return motif_order[1:-1]


def viterbi(seq, emission_mat, transition_mat, k_counter):
    """
    calculates the most probable motif location
    :param seq: sequence
    :param emission_mat
    :param transition_mat
    :param k_counter: num of states
    :return: seq of states, aligned to original seq
    """
    k_dim = k_counter + NOT_MOTIF_STATES
    N = len(seq)
    prob_mat = wrap_log(np.zeros([k_dim, N]))
    trace_mat = np.zeros([k_dim, N])
    prob_mat[0, 0] = wrap_log(1)
    for j in range(1, N):
        curr_letter = prob_mat[:, j - 1].reshape((-1, 1))
        potential_trans = curr_letter + transition_mat
        max_values = np.max(potential_trans, axis=0).T
        trace_mat[:, j] = np.argmax(potential_trans, axis=0).T
        prob_mat[:, j] = max_values + emission_mat[:, emission_dict[seq[j]]]
    # begin trace
    motif_order = EMPTY_STRING
    curr_k = int(np.argmax(prob_mat[:, -1]))
    for j in range(N - 1, -1, -1):
        last_motif_state = k_dim - STATES_AT_END
        if FIRST_MOTIF_STATE <= curr_k <= last_motif_state:
            motif_order = MOTIF + motif_order
        else:
            motif_order = BACKGROUND + motif_order
        curr_k = int(trace_mat[curr_k, j])
    return motif_order[1:-1]


def wrap_log(to_wrap):
    """
    helper func to avoid log warnings for np.log
    :param to_wrap: element to log
    :return: np.log(to_wrap)
    """
    with np.errstate(divide='ignore'):
        result = np.log(to_wrap)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    args = parser.parse_args()

    emission_mat, k_counter = parse_emission_matrix(args.initial_emission)
    transition_mat = build_transition_matrix(k_counter, args.q, args.p)
    original_seq = args.seq
    seq = START_SIGN + original_seq + END_SIGN

    if args.alg == 'viterbi':
        result = viterbi(seq, emission_mat, transition_mat, k_counter)
        print_results(original_seq, result)

    elif args.alg == 'forward':
        result = forward(seq, emission_mat, transition_mat, k_counter)[-1, -1]
        print(result)

    elif args.alg == 'backward':
        result = backward(seq, emission_mat, transition_mat, k_counter)[0, 0]
        print(result)

    elif args.alg == 'posterior':
        result = posterior(seq, emission_mat, transition_mat, k_counter)
        print_results(original_seq, result)


if __name__ == '__main__':
    main()
