import argparse
import random
import numpy as np
import pickle
from hmm import build_emission_matrices, build_initial_distribution, CustomHMM
from tqdm import tqdm
import os


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11111)
    parser.add_argument("--folder", type=str, default="")
    return parser.parse_args()


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)


def main():
    # init
    args = parse_arguments()
    set_seed(args.seed)

    # load saved A matrix
    DATA_PATH = 'data/' + args.folder
    file = open(os.path.join(DATA_PATH, "A_matrix.pickle"), 'rb')
    object_file = pickle.load(file)
    _num_states, _pi_0, _lambda2, _U, _Sigma, _U_inv, _A, _entropy = object_file

    _num_states = np.array(_num_states)
    _pi_0 = np.array(_pi_0)
    _lambda2 = np.array(_lambda2)
    _U = np.array(_U)
    _Sigma = np.array(_Sigma)
    _U_inv = np.array(_U_inv)
    _A = np.array(_A)
    _entropy = np.array(_entropy)

    # generate sequences
    MAX_SEQ_LEN = 2049
    ENTROPY_GAP = 1
    NUM_SEQUENCES = 4096

    num_states = []
    steady_states = []
    lambda2s = []
    Us = []
    Sigmas = []
    U_invs = []
    As = []
    A_entropys = []
    observations = []
    hidden_states = []
    pi_0s = []

    for idx, (num_state, A) in tqdm(enumerate(zip(_num_states, _As)), total=len(_As)):
        for observation_length in [4, 8, 16, 32, 64]:
            means_list, stds_list = build_emission_matrices(num_state, -float(obs_length) / 2, float(obs_length) / 2, std_gap=ENTROPY_GAP)
            pis = build_initial_distribution(num_state)[:1]
            for means, stds in zip(means_list, stds_list):
                for pi in pis:

                    num_states.append(num_state)
                    steady_states.append(_steady_states[idx])
                    lambda2s.append(_lambda2s[idx])
                    Us.append(_Us[idx])
                    Sigmas.append(_Sigmas[idx])
                    U_invs.append(_U_invs[idx])
                    As.append(A)
                    A_entropys.append(_A_entropys[idx])
                    pi_0s.append(pi)

                    # generate sequence
                    hmm = CustomHMM(np.arange(num_state), np.array(means), np.array(stds), np.array(A) / np.sum(A, axis=1, keepdims=True), np.array(pi))
                    observation, hidden_state = hmm.generate_dataset(NUM_SEQUENCES, MAX_SEQ_LEN, args.seed)
                    observations.append(np.array(observation).tolist())
                    hidden_states.append(np.array(hidden_state).tolist())

    DATA_PATH = 'data/'
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, 'generations.pickle'), 'wb') as f:
        pickle.dump((num_states, steady_states, lambda2s, Us, Sigmas, U_invs, As, A_entropys, observations, hidden_states, means_list, stds_list, pi_0s), f)


if __name__ == "__main__":
    main()