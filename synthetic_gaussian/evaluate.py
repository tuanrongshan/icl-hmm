import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import pickle
from functools import partial
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", type=str, default="data/generations.pickle")
    parser.add_argument("--save_pickle", type=str, default="syn_results.pickle")
    parser.add_argument("--save_csv", type=str, default="syn_results.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11111)
    parser.add_argument("--num_seq", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=8)
    return parser.parse_args()


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def number_to_index(n: int) -> str:
    result = ""
    m = n + 1
    while m > 0:
        m, remainder = divmod(m - 1, 26)
        result = chr(65 + remainder) + result
    return ' ' + result


def kl_divergence(logp, logq):
    p = torch.exp(logp)
    return (p * (logp - logq)).sum(-1)


def hellinger_distance(p, q):
    squared_hellinger = 1.0 - torch.sqrt(p * q).sum(-1)
    squared_hellinger = torch.clamp(squared_hellinger, min=0.0)
    return torch.sqrt(squared_hellinger)


def np_kl_divergence(logp, logq):
    p = np.exp(logp)
    return (p * (logp - logq)).sum(axis=-1)


def np_hellinger_distance(p, q):
    squared_hellinger = 1.0 - np.sqrt(p * q).sum(-1)
    squared_hellinger = np.clip(squared_hellinger, a_min=0.0, a_max=None)
    return np.sqrt(squared_hellinger)


def add_to_results(results, new_result):

    for k, v in new_result.items():
        if k in results:
            results[k].append(str(np.array(v).tolist()))
        else:
            results[k] = [str(np.array(v).tolist())]
    
    return results


def evaluate_llm(model, tokenizer, emission_prob, state_seq, emission_seq, index_map, batch_size, place_to_eval):

    # index to token
    index_token_map = []
    for k in index_map:
        token = tokenizer(k)['input_ids']
        assert len(token) == 1
        index_token_map.append(token[0])
    index_token_map = np.array(index_token_map)

    # construct prompt
    inputs = []
    for seq in emission_seq:
        inputs.append(index_token_map[seq])
    inputs = np.array(inputs)

    # prepare
    emission_prob = torch.from_numpy(emission_prob).to(model.device)
    emission_logprob = torch.log(emission_prob)
    index_token_map = torch.from_numpy(index_token_map).long().to(model.device)
    place_to_eval = torch.tensor(place_to_eval).long().to(model.device)

    # evaluate
    llm_emission_acc, llm_emission_prob, llm_emission_reverse_kl, llm_emission_forward_kl, llm_emission_hellinger_distance = [], [], [], [], []
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(inputs), batch_size)):

            # prepare batch
            end_idx = min(start_idx + batch_size, len(inputs))
            batch = torch.from_numpy(inputs[start_idx:end_idx]).long().to(model.device)

            batch_label = torch.tensor(emission_seq[start_idx:end_idx]).long().to(model.device)[:, place_to_eval]
            batch_state_label = torch.tensor(state_seq[start_idx:end_idx]).long().to(model.device)[:, place_to_eval]

            # gather prob
            output = model(batch, return_dict=True)
            logits = output.logits[:, place_to_eval - 1]
            all_logprob = F.log_softmax(logits, dim=-1)[:, :, index_token_map]
            all_prob = torch.exp(all_logprob)
            all_prob = all_prob / all_prob.sum(-1, keepdim=True)

            # compute accuracy
            predicted_emission = torch.argmax(all_prob, dim=-1)
            llm_emission_acc.append(predicted_emission == batch_label)

            # compute prob
            prob = torch.gather(all_prob, 2, batch_label.unsqueeze(-1)).squeeze(-1)
            llm_emission_prob.append(prob)

            # compute kl
            all_logprob = torch.log(all_prob)
            label_logprob_label = emission_logprob[batch_state_label]
            llm_emission_reverse_kl.append(kl_divergence(all_logprob, label_logprob_label))
            llm_emission_forward_kl.append(kl_divergence(label_logprob_label, all_logprob))
            llm_emission_hellinger_distance.append(hellinger_distance(all_prob, emission_prob[batch_state_label]))

    all_results = [
        torch.cat(llm_emission_acc).float().cpu().tolist(),
        torch.cat(llm_emission_prob).float().cpu().tolist(),
        torch.cat(llm_emission_reverse_kl).float().cpu().tolist(),
        torch.cat(llm_emission_forward_kl).float().cpu().tolist(),
        torch.cat(llm_emission_hellinger_distance).float().cpu().tolist(),
    ]

    llm_emission_acc = torch.cat(llm_emission_acc).float().mean(0).cpu().tolist()
    llm_emission_prob = torch.cat(llm_emission_prob).float().mean(0).cpu().tolist()
    llm_emission_reverse_kl = torch.cat(llm_emission_reverse_kl).float().mean(0).cpu().tolist()
    llm_emission_forward_kl = torch.cat(llm_emission_forward_kl).float().mean(0).cpu().tolist()
    llm_emission_hellinger_distance = torch.cat(llm_emission_hellinger_distance).float().mean(0).cpu().tolist()

    return {
        'llm_emission_acc': llm_emission_acc,
        'llm_emission_prob': llm_emission_prob,
        'llm_emission_reverse_kl': llm_emission_reverse_kl,
        'llm_emission_forward_kl': llm_emission_forward_kl,
        'llm_emission_hellinger_distance': llm_emission_hellinger_distance,
    }, all_results


def evaluate_random(emission_prob, state_seq, emission_seq, place_to_eval):

    # prep
    state_seq = np.array(state_seq).astype(int)
    emission_seq = np.array(emission_seq).astype(int)
    place_to_eval = np.array(place_to_eval).astype(int)

    label = emission_seq[:, place_to_eval]
    state_label = state_seq[:, place_to_eval]
    emission_prob_label = emission_prob[state_label]
    random_prob = np.full((1, 1, emission_prob.shape[1]), 1 / emission_prob.shape[1])

    # compute logprob
    emission_logprob_label = np.log(emission_prob_label)
    random_logprob = np.log(random_prob)

    # compute metrics
    random_emission_reverse_kl = np_kl_divergence(random_logprob, emission_logprob_label).mean(0)
    random_emission_forward_kl = np_kl_divergence(emission_logprob_label, random_logprob).mean(0)
    random_emission_hellinger_distance = np_hellinger_distance(random_prob, emission_prob_label).mean(0)

    return {
        'random_emission_acc': 1 / emission_prob.shape[1],
        'random_emission_prob': 1 / emission_prob.shape[1],
        'random_emission_reverse_kl': random_emission_reverse_kl,
        'random_emission_forward_kl': random_emission_forward_kl,
        'random_emission_hellinger_distance': random_emission_hellinger_distance,
    }


def baum_welch(observations, n_states, n_emissions, max_iter = 100, tol = 1e-6):
    T = len(observations)
    obs = np.array(observations)
    
    # Initialize parameters randomly (with normalization)
    A = np.random.rand(n_states, n_states)  # Transition probabilities
    A = A / A.sum(axis=1, keepdims=True)
    
    B = np.random.rand(n_states, n_emissions)  # Emission probabilities
    B = B / B.sum(axis=1, keepdims=True)
    
    pi = np.random.rand(n_states)  # Initial state probabilities
    pi = pi / pi.sum()
    
    log_likelihood_prev = -np.inf
    
    for iteration in range(max_iter):
        # Forward-Backward algorithm
        alpha, beta, scale_factors, log_likelihood = forward_backward(obs, A, B, pi)
        
        # Check for convergence
        if abs(log_likelihood - log_likelihood_prev) < tol:
            break
        log_likelihood_prev = log_likelihood
        
        # Compute expected state occupancy and transition counts
        gamma, xi = compute_expected_counts(alpha, beta, A, B, obs, scale_factors)
        
        # Re-estimate parameters
        pi = gamma[0, :].copy()
        
        # Re-estimate A (transition matrix)
        denominator = gamma[:T-1].sum(axis=0)
        numerator = xi.sum(axis=0)
        
        # Handle states that are never visited
        mask = denominator > 0
        A[mask] = (numerator / denominator[:, np.newaxis])[mask]
        A[~mask] = 1.0 / n_states  # Uniform distribution for unvisited states
        
        # Ensure each row sums to 1 (valid probability distribution)
        A = A / A.sum(axis=1, keepdims=True)
        
        # Re-estimate B (emission matrix)
        denominator = gamma.sum(axis=0)
        
        # For each emission value, create a binary mask
        for k in range(n_emissions):
            # Vectorized computation for each emission k
            mask_obs = (obs == k)
            
            # Compute numerator for emission k across all states
            mask_states = denominator > 0
            B[mask_states, k] = (gamma[mask_obs].sum(axis=0) / denominator)[mask_states]
            B[~mask_states, k] = 1.0 / n_emissions  # Uniform for unvisited states
        
        # Ensure each row of B sums to 1 (valid probability distribution)
        B = B / B.sum(axis=1, keepdims=True)

    # calculate the probs
    alpha = pi * B[:, obs[0]]
    alpha = alpha / np.sum(alpha)

    for t in range(1, len(obs)):
        alpha = (alpha.dot(A)) * B[:, obs[t]]
        alpha = alpha / np.sum(alpha)

    alpha = alpha / np.sum(alpha)
    next_state_prob = alpha.dot(A)
    next_obs_prob = next_state_prob.dot(B)

    return next_obs_prob


def forward_backward(obs, A, B, pi):
    T = len(obs)
    n_states = A.shape[0]
    
    # Initialize matrices
    alpha = np.zeros((T, n_states))
    beta = np.zeros((T, n_states))
    scale_factors = np.zeros(T)
    
    # Forward algorithm with scaling
    # Initialization
    alpha[0] = pi * B[:, obs[0]]
    scale_factors[0] = 1.0 / np.sum(alpha[0])
    alpha[0] *= scale_factors[0]
    
    # Recursion (vectorized)
    for t in range(1, T):
        # Matrix multiplication for efficient computation
        alpha[t] = np.dot(alpha[t-1], A) * B[:, obs[t]]
        
        # Scale to prevent underflow
        scale_factors[t] = 1.0 / np.sum(alpha[t])
        if not np.isfinite(scale_factors[t]):  # Handle potential division by zero
            scale_factors[t] = 1.0
        alpha[t] *= scale_factors[t]
    
    # Backward algorithm with scaling
    # Initialization
    beta[T-1] = 1.0 * scale_factors[T-1]
    
    # Recursion (vectorized)
    for t in range(T-2, -1, -1):
        # Vectorized computation using broadcasting
        beta[t] = np.sum(A * (B[:, obs[t+1]] * beta[t+1]), axis=1) * scale_factors[t]
    
    # Compute log likelihood from scaling factors
    log_likelihood = -np.sum(np.log(scale_factors))
    
    return alpha, beta, scale_factors, log_likelihood


def compute_expected_counts(alpha, beta, A, B, obs, scale_factors):
    T = len(obs)
    n_states = A.shape[0]
    
    # Compute gamma (expected state occupancy) - vectorized
    gamma = alpha * beta
    # Normalize each row
    gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
    
    # Compute xi (expected transition counts) - partially vectorized
    xi = np.zeros((T-1, n_states, n_states))
    
    # Pre-compute emission probabilities for observations
    b_obs = np.array([B[:, obs[t+1]] for t in range(T-1)])
    
    for t in range(T-1):
        # Matrix multiplication using outer product
        xi[t] = np.outer(alpha[t], beta[t+1] * b_obs[t]) * A
        # Normalize
        xi[t] /= np.sum(xi[t])
    
    return gamma, xi


def evaluate_bw(A, B, pi, state_seq, observations, place_to_eval):

    # prep
    state_seq = np.array(state_seq).astype(int)
    observations = np.array(observations).astype(int)
    place_to_eval = np.array(place_to_eval).astype(int)

    # compute metrics
    result = defaultdict(list)
    label = observations[:, place_to_eval]
    state_label = state_seq[:, place_to_eval]
    emission_prob_label = B[state_label]
    emission_logprob_label = np.log(emission_prob_label)

    baum_welch_partial = partial(baum_welch, n_states=A.shape[0], n_emissions=B.shape[1], max_iter=100, tol=1e-6)

    # compute for each place
    all_result = None
    for i in range(len(place_to_eval)):
        
        # compute metrics at the current place
        place = place_to_eval[i]
        place_label = label[:, i]
        place_emission_prob_label = emission_prob_label[:, i, :]
        place_emission_logprob_label = emission_logprob_label[:, i, :]
        histories = observations[:, :place]

        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(baum_welch_partial, histories), total=len(histories)))
        emission_prob = np.array(results)
        emission_logprob = np.log(emission_prob)

        # gather results
        bw_acc = np.argmax(emission_prob, axis=-1) == place_label
        bw_prob = emission_prob[np.arange(len(emission_prob)), place_label]
        bw_reverse_kl = np_kl_divergence(emission_logprob, place_emission_logprob_label)
        bw_forward_kl = np_kl_divergence(place_emission_logprob_label, emission_logprob)
        bw_hellinger_distance = np_hellinger_distance(emission_prob, place_emission_prob_label)

        if all_result == None:
            all_result = [[bw_acc], [bw_prob], [bw_reverse_kl], [bw_forward_kl], [bw_hellinger_distance]]
        else:
            all_result[0].append(bw_acc)
            all_result[1].append(bw_prob)
            all_result[2].append(bw_reverse_kl)
            all_result[3].append(bw_forward_kl)
            all_result[4].append(bw_hellinger_distance)

        # compute unigram
        result['bw_acc'].append(bw_acc.mean())
        result['bw_prob'].append(bw_prob.mean())
        result['bw_reverse_kl'].append(bw_reverse_kl.mean())
        result['bw_forward_kl'].append(bw_forward_kl.mean())
        result['bw_hellinger_distance'].append(bw_hellinger_distance.mean())

    for i in range(5):
        all_result[i] = np.stack(all_result[i], axis=1).tolist()

    return result, all_result

class AutoregressiveLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        x_embed = self.embedding(x)
        
        if hidden is None:
            lstm_out, hidden = self.lstm(x_embed)
        else:
            lstm_out, hidden = self.lstm(x_embed, hidden)

        logits = lstm_out[-1, :]
        logits = self.fc(logits)
        
        return logits, hidden
    
def train_eval_model(model, train_seq, label, epochs=1, lr=0.001):

    # init
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda")
    model = model.to(device)
    
    # train
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(train_seq)-1):
            
            x = train_seq[:i+1]
            target = train_seq[i+1]

            optimizer.zero_grad()
            logits, _ = model(x)

            loss = criterion(logits, target)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = torch.argmax(logits)
            correct_predictions += (pred == target).sum().item()
            total_predictions += 1
        
        # Calculate epoch metrics
        epoch_loss = total_loss / (len(train_seq) - 1)
        epoch_acc = correct_predictions / total_predictions
        
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # compute the logits of the next action
    model.eval()
    with torch.no_grad():
        logits, _ = model(train_seq)
    
    return logits


def evaluate_lstm(args, emission_prob, state_seq, emission_seq, num_observation, place_to_eval):

    # prepare
    emission_prob = torch.from_numpy(emission_prob).to("cuda")
    emission_logprob = torch.log(emission_prob)
    place_to_eval = torch.tensor(place_to_eval).long().to("cuda")

    # evaluate
    lstm_emission_acc, lstm_emission_prob, lstm_emission_reverse_kl, lstm_emission_forward_kl, lstm_emission_hellinger_distance = [], [], [], [], []
    for idx in tqdm(range(len(emission_seq))):
        all_logprob = []
        for place in place_to_eval:
            # prepare batch
            x = torch.from_numpy(emission_seq[idx][:place]).long().to("cuda")
            y = torch.from_numpy(emission_seq[idx][place:place+1]).long().to("cuda")

            # init model
            model = AutoregressiveLSTM(num_observation, args.embedding_dim, args.hidden_dim, args.num_layers)

            # train and evaluate model
            logits = train_eval_model(model, x, y, epochs=args.epochs, lr=args.lr)
            logprob = F.log_softmax(logits, dim=-1)
            all_logprob.append(logprob)

        # compute probs
        all_logprob = torch.stack(all_logprob)
        all_prob = torch.exp(all_logprob)
        all_prob = all_prob / all_prob.sum(-1, keepdim=True)

        # gather labels
        batch_label = torch.tensor(emission_seq[idx]).long().to("cuda")[place_to_eval]
        batch_state_label = torch.tensor(state_seq[idx]).long().to("cuda")[place_to_eval]

        # compute accuracy
        predicted_emission = torch.argmax(all_prob, dim=-1)
        lstm_emission_acc.append(predicted_emission == batch_label)

        # compute prob
        prob = torch.gather(all_prob, 1, batch_label.unsqueeze(-1)).squeeze(-1)
        lstm_emission_prob.append(prob)

        # compute kl
        all_logprob = torch.log(all_prob)
        label_logprob_label = emission_logprob[batch_state_label]
        lstm_emission_reverse_kl.append(kl_divergence(all_logprob, label_logprob_label))
        lstm_emission_forward_kl.append(kl_divergence(label_logprob_label, all_logprob))
        lstm_emission_hellinger_distance.append(hellinger_distance(all_prob, emission_prob[batch_state_label]))

    all_results = [
        torch.stack(lstm_emission_acc).float().cpu().tolist(),
        torch.stack(lstm_emission_prob).float().cpu().tolist(),
        torch.stack(lstm_emission_reverse_kl).float().cpu().tolist(),
        torch.stack(lstm_emission_forward_kl).float().cpu().tolist(),
        torch.stack(lstm_emission_hellinger_distance).float().cpu().tolist(),
    ]

    lstm_emission_acc = torch.stack(lstm_emission_acc).float().mean(0).cpu().tolist()
    lstm_emission_prob = torch.stack(lstm_emission_prob).float().mean(0).cpu().tolist()
    lstm_emission_reverse_kl = torch.stack(lstm_emission_reverse_kl).float().mean(0).cpu().tolist()
    lstm_emission_forward_kl = torch.stack(lstm_emission_forward_kl).float().mean(0).cpu().tolist()
    lstm_emission_hellinger_distance = torch.stack(lstm_emission_hellinger_distance).float().mean(0).cpu().tolist()

    return {
        'lstm_emission_acc': lstm_emission_acc,
        'lstm_emission_prob': lstm_emission_prob,
        'lstm_emission_reverse_kl': lstm_emission_reverse_kl,
        'lstm_emission_forward_kl': lstm_emission_forward_kl,
        'lstm_emission_hellinger_distance': lstm_emission_hellinger_distance,
    }, all_results


def safe_log(arr):
    """
    Compute log(arr) safely, mapping zeros to -infinity.
    """
    return np.where(arr > 0, np.log(arr), -np.inf)


def viterbi_vectorized(A, B, pi, state_seq, observations, place_to_eval):

    # prep
    state_seq = np.array(state_seq).astype(int)
    observations = np.array(observations).astype(int)
    place_to_eval = np.array(place_to_eval).astype(int)

    BATCH, T = observations.shape
    N = len(pi)

    # Compute log probabilities safely.
    logA = safe_log(A)      # Shape: (N, N)
    logB = safe_log(B)      # Shape: (N, num_observations)
    logpi = safe_log(pi)    # Shape: (N,)

    # Initialize the delta and psi tables.
    delta = np.zeros((BATCH, T, N))
    psi = np.zeros((BATCH, T, N), dtype=int)

    # Initialization at time t = 0.
    delta[:, 0, :] = logpi + logB[:, observations[:, 0]].T
    
    # Recursion: fill in delta and psi for t = 1, ..., T-1.
    for t in range(1, T):
        # candidate has shape (BATCH, N, N) where candidate[i, k, j] is the log-probability
        candidate = delta[:, t - 1, :][:, :, None] + logA[None, :, :]
        # For each sequence and state j, select the best previous state.
        delta[:, t, :] = candidate.max(axis=1) + logB[:, observations[:, t]].T
        psi[:, t, :] = candidate.argmax(axis=1)

    # compute metrics
    result = defaultdict(list)
    label = observations[:, place_to_eval]
    state_label = state_seq[:, place_to_eval]
    emission_prob_label = B[state_label]
    emission_logprob_label = np.log(emission_prob_label)

    # compute for each place
    for i in range(len(place_to_eval)):

        # compute metrics at the current place
        place = place_to_eval[i]
        place_label = label[:, i]
        place_emission_prob_label = emission_prob_label[:, i, :]
        place_emission_logprob_label = emission_logprob_label[:, i, :]

        # Backtrace to recover the most likely state sequence for each sequence.
        paths = np.zeros((BATCH, place), dtype=int)
        # Choose the best final state for each sequence.
        paths[:, place - 1] = delta[:, place - 1, :].argmax(axis=1)
        for t in range(place - 2, -1, -1):
            paths[:, t] = psi[:, t + 1, :][np.arange(BATCH), paths[:, t + 1]]

        # compute prob
        predicted_state = paths[:, place - 1]
        emission_prob = A[predicted_state, :] @ B
        emission_logprob = np.log(emission_prob)

        # compute unigram
        result['viterbi_acc'].append((np.argmax(emission_prob, axis=-1) == place_label).mean())
        result['viterbi_prob'].append(emission_prob[np.arange(BATCH), place_label].mean())
        result['viterbi_reverse_kl'].append(np_kl_divergence(emission_logprob, place_emission_logprob_label).mean())
        result['viterbi_forward_kl'].append(np_kl_divergence(place_emission_logprob_label, emission_logprob).mean())
        result['viterbi_hellinger_distance'].append(np_hellinger_distance(emission_prob, place_emission_prob_label).mean())

    return result


def evaluate_previous_prob(emission_prob, state_seq, emission_seq, place_to_eval):

    # prep
    state_seq = np.array(state_seq).astype(int)
    emission_seq = np.array(emission_seq).astype(int)
    place_to_eval = np.array(place_to_eval).astype(int)

    label = emission_seq[:, place_to_eval]
    state_label = state_seq[:, place_to_eval]
    emission_prob_label = emission_prob[state_label]

    V = emission_prob.shape[1]
    B = len(state_seq)
    b_idx = np.arange(B)[:, None]

    # compute logprob
    emission_logprob_label = np.log(emission_prob_label)

    # compute metrics
    result = defaultdict(list)
    
    # compute for each place
    for i in range(len(place_to_eval)):

        # compute metrics at the current place
        place = place_to_eval[i]
        place_label = label[:, i]
        place_emission_prob_label = emission_prob_label[:, i, :]
        place_emission_logprob_label = emission_logprob_label[:, i, :]
        histories = emission_seq[:, :place]

        # unigram_counts: B x V
        unigram_counts = np.zeros((B, V))
        np.add.at(unigram_counts, (b_idx, histories), 1)

        # bigram_counts: B x V x V
        bigram_counts = np.zeros((B, V, V))
        np.add.at(bigram_counts, (b_idx, histories[:, :-1], histories[:, 1:]), 1)

        # trigram_counts: B x V x V x V
        trigram_counts = np.zeros((B, V, V, V))
        np.add.at(trigram_counts, (b_idx, histories[:, :-2], histories[:, 1:-1], histories[:, 2:]), 1)

        # trigram_counts: B x V x V x V
        fourgram_counts = np.zeros((B, V, V, V, V))
        np.add.at(fourgram_counts, (b_idx, histories[:, :-3], histories[:, 1:-2], histories[:, 2:-1], histories[:, 3:]), 1)

        # compute prob
        dist1 = unigram_counts[b_idx[:, 0], :]
        dist2 = bigram_counts[b_idx[:, 0], histories[:, -1], :]
        dist3 = trigram_counts[b_idx[:, 0], histories[:, -2], histories[:, -1], :]
        dist4 = fourgram_counts[b_idx[:, 0], histories[:, -3], histories[:, -2], histories[:, -1], :]

        s1 = np.sum(dist1, axis=1)
        s2 = np.sum(dist2, axis=1)
        s3 = np.sum(dist3, axis=1)
        s4 = np.sum(dist4, axis=1)

        # create distribution for 4-gram
        mask4 = s4 > 0
        mask3 = (s4 == 0) & (s3 > 0)
        mask2 = (s4 == 0) & (s3 == 0) & (s2 > 0)
        mask0 = (s4 == 0) & (s3 == 0) & (s2 == 0)
        dist4[mask4] = dist4[mask4] / s4[mask4, None]
        dist4[mask3] = dist3[mask3] / s3[mask3, None]
        dist4[mask2] = dist2[mask2] / s2[mask2, None]
        dist4[mask0] = 1.0 / V

        # create distribution for 3-gram
        mask3 = (s3 > 0)
        mask2 = (s3 == 0) & (s2 > 0)
        mask0 = (s3 == 0) & (s2 == 0)
        dist3[mask3] = dist3[mask3] / s3[mask3, None]
        dist3[mask2] = dist2[mask2] / s2[mask2, None]
        dist3[mask0] = 1.0 / V

        # create distribution for 2-gram
        mask2 = (s2 > 0)
        mask0 = (s2 == 0)
        dist2[mask2] = dist2[mask2] / s2[mask2, None]
        dist2[mask0] = 1.0 / V

        # create distribution for 1-gram
        dist1 = dist1 / s1[:, None]

        # compute logprob
        log_dist4 = np.log(dist4)
        log_dist3 = np.log(dist3)
        log_dist2 = np.log(dist2)
        log_dist1 = np.log(dist1)

        # compute unigram
        result['1-gram_acc'].append((np.argmax(dist1, axis=-1) == place_label).mean())
        result['1-gram_prob'].append(dist1[b_idx[:, 0], place_label].mean())
        result['1-gram_reverse_kl'].append(np_kl_divergence(log_dist1, place_emission_logprob_label).mean())
        result['1-gram_forward_kl'].append(np_kl_divergence(place_emission_logprob_label, log_dist1).mean())
        result['1-gram_hellinger_distance'].append(np_hellinger_distance(dist1, place_emission_prob_label).mean())

        # compute bigram
        result['2-gram_acc'].append((np.argmax(dist2, axis=-1) == place_label).mean())
        result['2-gram_prob'].append(dist2[b_idx[:, 0], place_label].mean())
        result['2-gram_reverse_kl'].append(np_kl_divergence(log_dist2, place_emission_logprob_label).mean())
        result['2-gram_forward_kl'].append(np_kl_divergence(place_emission_logprob_label, log_dist2).mean())
        result['2-gram_hellinger_distance'].append(np_hellinger_distance(dist2, place_emission_prob_label).mean())

        # compute trigram
        result['3-gram_acc'].append((np.argmax(dist3, axis=-1) == place_label).mean())
        result['3-gram_prob'].append(dist3[b_idx[:, 0], place_label].mean())
        result['3-gram_reverse_kl'].append(np_kl_divergence(log_dist3, place_emission_logprob_label).mean())
        result['3-gram_forward_kl'].append(np_kl_divergence(place_emission_logprob_label, log_dist3).mean())
        result['3-gram_hellinger_distance'].append(np_hellinger_distance(dist3, place_emission_prob_label).mean())

        # compute 4-gram
        result['4-gram_acc'].append((np.argmax(dist4, axis=-1) == place_label).mean())
        result['4-gram_prob'].append(dist4[b_idx[:, 0], place_label].mean())
        result['4-gram_reverse_kl'].append(np_kl_divergence(log_dist4, place_emission_logprob_label).mean())
        result['4-gram_forward_kl'].append(np_kl_divergence(place_emission_logprob_label, log_dist4).mean())
        result['4-gram_hellinger_distance'].append(np_hellinger_distance(dist4, place_emission_prob_label).mean())
    
    return result


def p_o_t_given_h_orcale(A, B, h_t, emission_label, emission_prob_label, emission_logprob_label):

    p_dot_h_t = A[h_t, :]
    p_o_t_given_h = p_dot_h_t @ B
    logp_o_t_given_h = np.log(p_o_t_given_h)

    # compute accuracy
    p_o_t_given_h_acc = (np.argmax(p_o_t_given_h, axis=-1) == emission_label).mean()

    # compute prob
    p_o_t_given_h_prob = p_o_t_given_h[np.arange(len(p_o_t_given_h)), emission_label].mean()

    # compute kl
    p_o_t_given_h_reverse_kl = np_kl_divergence(logp_o_t_given_h, emission_logprob_label).mean()
    p_o_t_given_h_forward_kl = np_kl_divergence(emission_logprob_label, logp_o_t_given_h).mean()
    p_o_t_given_h_hellinger_distance = np_hellinger_distance(p_o_t_given_h, emission_prob_label).mean()

    return {
        'p_o_given_prev_h_acc': p_o_t_given_h_acc,
        'p_o_given_prev_h_prob': p_o_t_given_h_prob,
        'p_o_given_prev_h_reverse_kl': p_o_t_given_h_reverse_kl,
        'p_o_given_prev_h_forward_kl': p_o_t_given_h_forward_kl,
        'p_o_given_prev_h_hellinger_distance': p_o_t_given_h_hellinger_distance,
    }


def p_o_t_given_prev_k_o_orcale(A, B, pi, history, t, emission_label, emission_prob_label, emission_logprob_label, logk=None):

    # record k
    k = history.shape[-1]

    # calculate the probs
    p_h_t_k = np.linalg.matrix_power(A, t - k) @ np.expand_dims(pi, axis=1) # p(h_{t-k})

    alpha = p_h_t_k * B[:, history[:, 0]]
    alpha = alpha / np.sum(alpha, axis=0)

    for i in range(1, k):
        alpha = (A @ alpha) * B[:, history[:, i]]
        alpha = alpha / np.sum(alpha, axis=0)

    p_h_t = alpha
    p_h_next = A @ p_h_t
    next_emission_prob = p_h_next.T @ B
    next_emission_logprob = np.log(next_emission_prob)

    # compute accuracy
    p_o_t_given_prev_k_acc = (np.argmax(next_emission_prob, axis=-1) == emission_label).mean()

    # compute prob
    p_o_t_given_prev_k_prob = next_emission_prob[np.arange(len(next_emission_prob)), emission_label].mean()

    # compute kl
    p_o_t_given_prev_k_reverse_kl = np_kl_divergence(next_emission_logprob, emission_logprob_label).mean()
    p_o_t_given_prev_k_forward_kl = np_kl_divergence(emission_logprob_label, next_emission_logprob).mean()
    p_o_t_given_prev_k_hellinger_distance = np_hellinger_distance(next_emission_prob, emission_prob_label).mean()

    if logk == None:
        return {
            f'p_o_t_given_prev_{k}_o_acc': p_o_t_given_prev_k_acc,
            f'p_o_t_given_prev_{k}_o_prob': p_o_t_given_prev_k_prob,
            f'p_o_t_given_prev_{k}_o_reverse_kl': p_o_t_given_prev_k_reverse_kl,
            f'p_o_t_given_prev_{k}_o_forward_kl': p_o_t_given_prev_k_forward_kl,
            f'p_o_t_given_prev_{k}_o_hellinger_distance': p_o_t_given_prev_k_hellinger_distance,
        }
    else:
        return {
            f'p_o_t_given_prev_{logk}_o_acc': p_o_t_given_prev_k_acc,
            f'p_o_t_given_prev_{logk}_o_prob': p_o_t_given_prev_k_prob,
            f'p_o_t_given_prev_{logk}_o_reverse_kl': p_o_t_given_prev_k_reverse_kl,
            f'p_o_t_given_prev_{logk}_o_forward_kl': p_o_t_given_prev_k_forward_kl,
            f'p_o_t_given_prev_{logk}_o_hellinger_distance': p_o_t_given_prev_k_hellinger_distance,
        }


def concat_result(result, new_result):
    for k, v in new_result.items():
        if k in result:
            result[k].append(v)
        else:
            result[k] = [v]
    return result


def evaluate_oracle_result(transition_prob, emission_prob, initial_prob, state_seq, emission_seq, place_to_eval, prev_k):

    # prep
    state_seq = np.array(state_seq).astype(int)
    emission_seq = np.array(emission_seq).astype(int)
    place_to_eval = np.array(place_to_eval).astype(int)

    label = emission_seq[:, place_to_eval]
    state_label = state_seq[:, place_to_eval]
    emission_prob_label = emission_prob[state_label]
    emission_logprob_label = np.log(emission_prob_label)

    # result
    result = {}
    
    # compute for each place
    for i in range(len(place_to_eval)):

        # compute metrics at the current place
        place = place_to_eval[i]
        place_label = label[:, i]
        place_emission_prob_label = emission_prob_label[:, i, :]
        place_emission_logprob_label = emission_logprob_label[:, i, :]

        # baseline for p(o_t+1|h_t)
        result = concat_result(result, p_o_t_given_h_orcale(transition_prob, emission_prob, state_seq[:, place-1], place_label, place_emission_prob_label, place_emission_logprob_label))

        # baseline for p(o_t+1|o_t-k:t)
        histories = emission_seq[:, :place]
        for k in range(1, prev_k+1):
            result = concat_result(result, p_o_t_given_prev_k_o_orcale(transition_prob, emission_prob, initial_prob, histories[:, -k:], place, place_label, place_emission_prob_label, place_emission_logprob_label))

        # baseline for p(o_t+1|o_0:t)
        result = concat_result(result, p_o_t_given_prev_k_o_orcale(transition_prob, emission_prob, initial_prob, histories, place, place_label, place_emission_prob_label, place_emission_logprob_label, logk='all'))

    return result


def main():

    # init
    args = parse_arguments()
    set_seed(args.seed)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")

    MAX_NUM_OBSERVATIONS = 64
    SEQ_LENGTH = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    BASELINE_PREV_K = 4

    with open(args.dataset, 'rb') as file:
        object_file = pickle.load(file)
    num_states, steady_states, lambda2s, Us, Sigmas, U_invs, As, A_entropys, num_observations, observations, hidden_states, Bs, B_entropys, pi_0s = object_file
    
    # prep for evaluation
    assert MAX_NUM_OBSERVATIONS < 26 ** 2

    # gather string index
    all_index_map = []
    i = -1
    while len(all_index_map) < MAX_NUM_OBSERVATIONS:
        i += 1
        if len(tokenizer(number_to_index(i))['input_ids']) != 1:
            continue
        all_index_map.append(number_to_index(i))
    all_index_map = np.array(all_index_map)

    # evaluation
    results = {}
    llm_all_results, bw_all_results, lstm_all_results = [], [], []
    for num_state, steady_state, lambda2, U, Sigma, U_inv, A, A_entropy, num_observation, observation, hidden_state, B, B_entropy, pi_0 in tqdm(zip(num_states, steady_states, lambda2s, Us, Sigmas, U_invs, As, A_entropys, num_observations, observations, hidden_states, Bs, B_entropys, pi_0s), total=len(num_states)):

        A = (np.array(A) / np.sum(A, axis=1, keepdims=True)).tolist()

        # record meta info
        meta_info = {}
        meta_info['num_state'] = num_state
        meta_info['steady_state'] = steady_state
        meta_info['lambda2'] = lambda2
        meta_info['U'] = U
        meta_info['Sigma'] = Sigma
        meta_info['U_inv'] = U_inv
        meta_info['A'] = A
        meta_info['A_entropy'] = A_entropy
        meta_info['num_observation'] = num_observation
        meta_info['B'] = B
        meta_info['B_entropy'] = B_entropy
        meta_info['pi_0'] = pi_0
        results = add_to_results(results, meta_info)

        # prep
        A = np.array(A)
        B = np.array(B)
        pi_0 = np.array(pi_0)

        # record llm_result
        llm_result, llm_all_result = evaluate_llm(model, tokenizer, B, hidden_state, observation, all_index_map[:num_observation], args.batch_size, SEQ_LENGTH)
        results = add_to_results(results, llm_result)
        llm_all_results.append(llm_all_result)
        print('done LLM')
        
        # record random_result
        random_result = evaluate_random(B, hidden_state, observation, SEQ_LENGTH)
        results = add_to_results(results, random_result)
        print('done Random')

        # record previous_prob_result
        previous_prob_result = evaluate_previous_prob(B, hidden_state, observation, SEQ_LENGTH)
        results = add_to_results(results, previous_prob_result)
        print('done previous prob')

        # record oracle_result
        oracle_result = evaluate_oracle_result(A, B, pi_0, hidden_state, observation, SEQ_LENGTH, BASELINE_PREV_K)
        results = add_to_results(results, oracle_result)
        print('done oracle')

        # viterbi algorithm
        viterbi_result = viterbi_vectorized(A, B, pi_0, hidden_state, observation, SEQ_LENGTH)
        results = add_to_results(results, viterbi_result)
        print('done Viterbi')

        # BW algorithm
        bw_result, bw_all_result = evaluate_bw(A, B, pi_0, hidden_state, observation, SEQ_LENGTH)
        results = add_to_results(results, bw_result)
        bw_all_results.append(bw_all_result)
        print('done BW')

        # record lstm_result
        lstm_result, lstm_all_result = evaluate_lstm(args, B, hidden_state, observation, num_observation, SEQ_LENGTH)
        results = add_to_results(results, lstm_result)
        lstm_all_results.append(lstm_all_result)
        print('done lstm')

    # Save the results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(args.save_csv, index=False)

    # save llm_all_results, bw_all_results, and lstm_all_results
    with open(args.save_pickle, 'wb') as f:
        pickle.dump((llm_all_results, bw_all_results, lstm_all_results), f)


if __name__ == "__main__":
    main()