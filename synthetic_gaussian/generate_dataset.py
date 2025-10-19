"""
Generate 100 seq (len=100) for each of (204 A matrices, 7 emission matrices)
Observations: [-64, 64]
"""
import argparse
import random
import numpy as np
import pickle
from hmm import build_emission_matrices_std, build_initial_distribution, CustomHMM
from tqdm import tqdm
import os


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11111)
    parser.add_argument("--store_folder", type=str, default="")
    return parser.parse_args()


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)


def main():
    # init
    args = parse_arguments()
    set_seed(args.seed)

    # load saved A matrix
    DATA_PATH = 'data/'
    file = open(os.path.join(DATA_PATH, "A_matrix.pickle"), 'rb')
    object_file = pickle.load(file)
    _num_states, _pi_0, _lambda2, _U, _Sigma, _U_inv, _A, _entropy = object_file

    _num_states = np.array(_num_states)

    # generate sequences
    MAX_SEQ_LEN = 100
    NUM_SEQUENCES = 100
    OBS_LENGTH = 128

    num_states = []
    lambda2s = []
    Us = []
    Sigmas = []
    U_invs = []
    As = []
    A_entropys = []
    observations = []
    hidden_states = []
    means_list = []
    stds_list = []
    pi_0s = []

    for entropy in range(7):
        for idx, (num_state, A) in tqdm(enumerate(zip(_num_states, _A)), total=len(_A)):
            means, stds = build_emission_matrices_std(num_state, -float(OBS_LENGTH) / 2, float(OBS_LENGTH) / 2, float(entropy))
            pi = build_initial_distribution(num_state)[0]
            num_states += [num_state] * 100
            lambda2s += [_lambda2[idx]] * 100
            Us += [_U[idx]] * 100
            Sigmas += [_Sigma[idx]] * 100
            U_invs += [_U_inv[idx]] * 100
            As += [A] * 100
            A_entropys += [_entropy[idx]] * 100
            means_list += [means] * 100
            stds_list += [stds] * 100
            pi_0s += [pi] * 100

            # generate sequence
            hmm = CustomHMM(np.arange(num_state), np.array(means), np.array(stds), np.array(A) / np.sum(A, axis=1, keepdims=True), np.array(pi))
            observation, hidden_state = hmm.generate_dataset(NUM_SEQUENCES, MAX_SEQ_LEN, args.seed)
            observations += np.array(observation).tolist()
            hidden_states += np.array(hidden_state).tolist()

    DATA_PATH = 'data/'
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, 'generations.pickle'), 'wb') as f:
        pickle.dump((num_states, lambda2s, Us, Sigmas, U_invs, As, A_entropys, observations, hidden_states, means_list, stds_list, pi_0s), f)


if __name__ == "__main__":
    main()