import numpy as np
import math
from typing import List, Tuple
from scipy.optimize import fsolve, minimize
from concurrent.futures import ProcessPoolExecutor

class HMM:
    def __init__(self):
        '''
        A is a matrix of size num_states x num_states
        B is a matrix of size num_states x num_observations
        pi is a vector of size num_states
        '''
        self.states = []
        self.num_states = len(self.states)
        self.observations = []
        self.num_observations = len(self.observations)
        self.A = np.array([])
        self.B = np.array([])
        self.pi = np.array([])

    def generate_sequence(self, length: int) -> Tuple[List[str], List[str]]:
        """
        Generate a sequence of observations and corresponding hidden states.
        
        Args:
            length: Length of the sequence to generate
            
        Returns:
            Tuple of (observations, hidden_states)
        """
        # Choose initial state based on pi
        current_state = np.random.choice(range(self.num_states), p=self.pi)
        
        hidden_states = [self.states[current_state]]
        observations = []
        
        for t in range(length):
            # Generate observation based on current state
            obs_idx = np.random.choice(range(self.num_observations), p=self.B[current_state])
            observations.append(self.observations[obs_idx])
            
            if t < length - 1:  # Don't transition after the last observation
                # Transition to next state
                current_state = np.random.choice(range(self.num_states), p=self.A[current_state])
                hidden_states.append(self.states[current_state])
        
        return observations, hidden_states

    def log_sum_exp(self, log_probs):
        """
        Calculate log(sum(exp(log_probs))) in a numerically stable way.
        """
        max_val = np.max(log_probs)
        return max_val + np.log(np.sum(np.exp(log_probs - max_val)))
    
    def viterbi_decode(self, observations: List[str]) -> List[str]:
        """
        Decode the most likely sequence of hidden states using the Viterbi algorithm.
        
        Args:
            observations: List of observations
            
        Returns:
            Most likely sequence of weather states
        """
        T = len(observations)
        
        # Convert observations to indices
        obs_indices = [self.observations.index(obs) for obs in observations]
        
        # Initialize Viterbi variables (using log probabilities)
        log_delta = np.zeros((self.num_states, T))
        psi = np.zeros((self.num_states, T), dtype=int)
        
        # Initialization step
        for s in range(self.num_states):
            log_delta[s, 0] = np.log(self.pi[s]) + np.log(self.B[s, obs_indices[0]])
            psi[s, 0] = 0
        
        # Recursion step
        for t in range(1, T):
            for s in range(self.num_states):
                # For each current state, find the most likely previous state
                log_probs = np.zeros(self.num_states)
                for s_prev in range(self.num_states):
                    log_probs[s_prev] = log_delta[s_prev, t-1] + np.log(self.A[s_prev, s])
                
                # Find the most likely previous state
                psi[s, t] = np.argmax(log_probs)
                log_delta[s, t] = log_probs[psi[s, t]] + np.log(self.B[s, obs_indices[t]])
        
        # Termination step: find the most likely final state
        final_state = np.argmax(log_delta[:, T-1])
        
        # Backtracking
        best_path = [final_state]
        for t in range(T-1, 0, -1):
            best_path.insert(0, psi[best_path[0], t])
        
        # Convert state indices to state names
        return [self.states[i] for i in best_path]
    
    def forward_algorithm(self, observations: List[str]) -> float:
        """
        Calculate the likelihood of observations using the Forward algorithm.
        
        Args:
            observations: List of observations
            
        Returns:
            Log probability of the observation sequence
        """
        T = len(observations)
        
        # Convert observations to indices
        obs_indices = [self.observations.index(obs) for obs in observations]
        
        # Initialize forward variables (in log space)
        log_alpha = np.zeros((self.num_states, T))
        
        # Initialization step
        for s in range(self.num_states):
            log_alpha[s, 0] = np.log(self.pi[s]) + np.log(self.B[s, obs_indices[0]])
        
        # Recursion step
        for t in range(1, T):
            for s in range(self.num_states):
                # Compute the log probability of transitioning to state s from all previous states
                log_probs = np.zeros(self.num_states)
                for s_prev in range(self.num_states):
                    log_probs[s_prev] = log_alpha[s_prev, t-1] + np.log(self.A[s_prev, s])
                
                # Sum these probabilities (in log space) and add the emission probability
                log_alpha[s, t] = self.log_sum_exp(log_probs) + np.log(self.B[s, obs_indices[t]])
        
        # Termination step: compute the log probability of the entire sequence
        return self.log_sum_exp(log_alpha[:, T-1])
    
    def backward_algorithm(self, observations: List[str]) -> np.ndarray:
        """
        Implement the Backward algorithm for HMM.
        
        Args:
            observations: List of observations
            
        Returns:
            Log-scaled backward variables
        """
        T = len(observations)
        
        # Convert observations to indices
        obs_indices = [self.observations.index(obs) for obs in observations]
        
        # Initialize backward variables (in log space)
        log_beta = np.zeros((self.num_states, T))
        
        # Initialization for the last time step (all states have beta = 1, so log(beta) = 0)
        for s in range(self.num_states):
            log_beta[s, T-1] = 0.0
        
        # Recursion step (going backward in time)
        for t in range(T-2, -1, -1):
            for s in range(self.num_states):
                log_probs = np.zeros(self.num_states)
                for s_next in range(self.num_states):
                    log_probs[s_next] = np.log(self.A[s, s_next]) + \
                                        np.log(self.B[s_next, obs_indices[t+1]]) + \
                                        log_beta[s_next, t+1]
                
                log_beta[s, t] = self.log_sum_exp(log_probs)
        
        return log_beta

    def probability_of_state_given_observations(self, observations: List[str]) -> List[List[float]]:
        """
        Calculate P(X_t = i | Y_{1:T}) for all states i and times t.
        
        Args:
            observations: List of umbrella observations
            
        Returns:
            List of probability distributions over states for each time step
        """
        T = len(observations)
        
        # Run forward and backward algorithms
        log_alpha = np.zeros((self.num_states, T))
        obs_indices = [self.observations.index(obs) for obs in observations]
        
        # Forward initialization
        for s in range(self.num_states):
            log_alpha[s, 0] = np.log(self.pi[s]) + np.log(self.B[s, obs_indices[0]])
        
        # Forward recursion
        for t in range(1, T):
            for s in range(self.num_states):
                log_probs = np.zeros(self.num_states)
                for s_prev in range(self.num_states):
                    log_probs[s_prev] = log_alpha[s_prev, t-1] + np.log(self.A[s_prev, s])
                
                log_alpha[s, t] = self.log_sum_exp(log_probs) + np.log(self.B[s, obs_indices[t]])
        
        # Run backward algorithm
        log_beta = self.backward_algorithm(observations)
        
        # Compute log P(O)
        log_p_o = self.log_sum_exp(log_alpha[:, T-1])
        
        # Compute P(X_t = i | Y_{1:T}) for all t and i
        state_probs = []
        for t in range(T):
            # Calculate unnormalized log probabilities
            log_probs = log_alpha[:, t] + log_beta[:, t] - log_p_o
            
            # Convert to probabilities and normalize
            probs = np.exp(log_probs)
            probs = probs / np.sum(probs)  # Ensure they sum to 1
            
            state_probs.append(probs)
        
        return state_probs

class CustomHMM(HMM):
    def __init__(self, states: List[int], observations: List[int], A: np.ndarray, B: np.ndarray, pi: np.ndarray):
        self.states = states
        self.num_states = len(states)
        self.observations = observations
        self.num_observations = len(observations)
        assert A.shape == (self.num_states, self.num_states)
        assert B.shape == (self.num_states, self.num_observations)
        assert pi.shape == (self.num_states,)
        self.A = A
        self.B = B
        self.pi = pi

    def generate_sequence(self, length: int, seed: int) -> Tuple[List[str], List[str]]:
        """
        Generate a sequence of observations and corresponding hidden states.
        
        Args:
            length: Length of the sequence to generate
            
        Returns:
            Tuple of (observations, hidden_states)
        """
        np.random.seed(seed)

        # Choose initial state based on pi
        current_state = np.random.choice(range(self.num_states), p=self.pi)
        
        hidden_states = [self.states[current_state]]
        observations = []
        
        for t in range(length):
            # Generate observation based on current state
            obs_idx = np.random.choice(range(self.num_observations), p=self.B[current_state])
            observations.append(self.observations[obs_idx])
            
            if t < length - 1:  # Don't transition after the last observation
                # Transition to next state
                current_state = np.random.choice(range(self.num_states), p=self.A[current_state])
                hidden_states.append(self.states[current_state])
        
        return observations, hidden_states

    def generate_dataset(self, num_sequences: int, length: int, seed: int) -> List[Tuple[List[str], List[str]]]:
        """
        Generate a dataset of sequences with varying lengths.
        
        Args:
            num_sequences: Number of sequences to generate
            length: Length of each sequence
            
        Returns:
            List of inputs, labels, input hidden states, label hidden states
        """
        with ProcessPoolExecutor() as executor:
            # Submit all tasks in parallel
            futures = [executor.submit(self.generate_sequence, length, n + seed) for n in range(num_sequences)]
            # Wait for all tasks to complete and gather the results
            results = [future.result() for future in futures]

        # Separate the observations and hidden states from the results
        observations, hidden_states = zip(*results)
        return list(observations), list(hidden_states)

def entropy_equation(p, n, e):
    # Avoid log(0) errors
    if p <= 0 or p >= 1 or n <= 1:
        return float('inf')
    
    term1 = -p * math.log2(p)
    term2 = -(1-p) * math.log2((1-p)/(n-1))
    
    # We want this to be 0 at the solution
    return term1 + term2 - e

def solve_for_p(n, e):
    if n <= 1:
        return "Error: n must be greater than 1"
    
    # Initial guess - try starting in the middle
    initial_p = 0.9
    
    # Solve the equation (find where it equals 0)
    solution = fsolve(lambda p: entropy_equation(p, n, e), initial_p)
    
    return solution[0]

def build_transition_matrices(num_states: int, entropy_gap: int = 0.5) -> List[np.ndarray]:

    # entropy = 0, deterministic
    A_1 = np.zeros((num_states, num_states))
    for i in range(num_states):
        A_1[i, (i+1)%num_states] = 1

    A = [A_1, np.eye(num_states)]
    e = entropy_gap
    while e < math.log2(num_states)+entropy_gap:
        p = solve_for_p(num_states, e)
        assert abs(-p * math.log2(p) - (1 - p) * math.log2((1 - p) / (num_states - 1)) - e) < 1e-5 and p >= (1 - p) / (num_states - 1) - 1e-5
        A.append(p * np.eye(num_states) + np.full((num_states, num_states), (1 - p) / (num_states - 1)) - np.eye(num_states) * (1 - p) / (num_states - 1))
        e += entropy_gap
    
    return A

def build_emission_matrices(num_states: int, num_observations: int, entropy_gap: int = 0.5) -> List[np.ndarray]:
    
    # entropy = 0
    B_0 = np.zeros((num_states, num_observations))
    for i in range(num_states):
        B_0[i, i%num_observations] = 1

    B = [B_0]
    entropies = [0]
    e = entropy_gap
    while e < math.log2(num_observations):
        p = solve_for_p(num_observations, e)
        assert abs(-p * math.log2(p) - (1 - p) * math.log2((1 - p) / (num_observations - 1)) - e) < 1e-5 and p >= (1 - p) / (num_observations - 1) - 1e-5
        B.append(p * B[0] + (1 - p) / (num_observations - 1) * (1 - B[0]))
        entropies.append(e)
        e += entropy_gap
    
    return B, entropies

def build_initial_distribution(num_states: int) -> List[np.ndarray]:
    
    # deterministic
    pi_deterministic = np.zeros(num_states)
    pi_deterministic[0] = 1

    # uniform
    pi_uniform = np.ones(num_states) / num_states

    return [pi_deterministic, pi_uniform]

def build_skew_static_distributions(num_states: int, num_distributions: int) -> List[np.ndarray]:
    """
    num_states: dimension of the probability vector
    num_distributions: number of distributions to generate, from uniform to concentrated
    """
    distributions = []
    # choose alphas exponentially from 0.1 to 100
    alphas = np.logspace(0.1, 2, num_distributions)
    for alpha in alphas:
        distributions.append(np.random.dirichlet(np.ones(num_states) * alpha, 1))
    return distributions

def calculate_mixing_rate(second_largest_eigenvalue: float) -> float:
    return 1/(1 - second_largest_eigenvalue) 

def find_orthonormal_vector(v: np.ndarray) -> np.ndarray:
    """
    Find a set of orthonormal vectors to the given vector v.
    
    Parameters:
    v (array-like): Input vector
    
    Returns:
    ndarray: Matrix whose columns form an orthonormal basis with normalized v
    """
    # Convert to numpy array and ensure it's a column vector
    v = np.asarray(v).reshape(-1, 1)
    
    # Normalize the vector
    v_normalized = v / np.linalg.norm(v)
    
    # Get the dimension
    n = len(v)
    
    # Create a matrix with v as a row
    A = v_normalized.T
    
    # Find the null space
    _, _, vh = np.linalg.svd(A)
    null_space = vh[1:].T  # Transpose to get column vectors
    
    # Combine v_normalized with its orthonormal vectors
    basis = np.hstack([v_normalized, null_space])
    
    return basis

def generate_linearly_independent_vectors(vector: np.ndarray, num_vectors: int) -> List[np.ndarray]:
    """
    Generate a set of linearly independent vectors to the given vector.
    """
    basis = find_orthonormal_vector(vector)
    return [basis[i] for i in range(num_vectors)]

def construct_U_simpler(pi_vector, max_attempts=100):
    """
    Constructs a matrix U where:
    - First column is all ones
    - Each column is linearly independent
    - The first row of U⁻¹ is pi_vector
    
    Uses random elements while maintaining the required properties.
    
    Parameters:
    pi_vector (numpy.ndarray): Input vector π
    max_attempts (int): Maximum number of attempts to generate a valid matrix
    
    Returns:
    numpy.ndarray: The constructed matrix U with random elements
    """
    n = pi_vector.shape[1]
    
    for attempt in range(max_attempts):
        # Initialize the matrix U with ones in the first column
        U = np.zeros((n, n))
        U[:, 0] = 1
        
        # Create random linearly independent columns for the rest of the matrix
        # while ensuring π·U = [1, 0, 0, ..., 0]
        
        # Generate random values for all rows except the first
        for j in range(1, n):
            # Use random normal distribution for more diversity in values
            U[1:, j] = np.random.randn(n-1) * (1 + np.random.rand())  # Scale for variety

            # Calculate the first row entry to ensure π·U_j = 0
            # This ensures the orthogonality condition: π·U_j = 0 for j ≥ 2
            U[0, j] = -np.sum(pi_vector[0, 1:] * U[1:, j]) / pi_vector[0, 0]
        
        # Verify columns are linearly independent
        if np.linalg.matrix_rank(U) == n:
            return U
    
    # If max attempts reached without success, use a more reliable approach
    # This fallback method is more deterministic but still produces a valid matrix
    print("Warning: Using the simplest method for U.")
    U = np.zeros((n, n))
    U[:, 0] = 1
    
    # Create an identity matrix for the rest with modified first row
    for j in range(1, n):
        U[j, j] = 1  # Diagonal elements
        U[0, j] = -pi_vector[j] / pi_vector[0]  # First row elements to satisfy π·U_j = 0
    
    return U

def construct_U(pi_vector, max_attempts=100):
    """
    An alternative approach that creates a highly random matrix U
    with the required properties.
    
    Parameters:
    pi_vector (numpy.ndarray): Input vector π
    max_attempts (int): Maximum number of attempts to generate a valid matrix
    
    Returns:
    numpy.ndarray: The constructed matrix U with more random structure
    """
    n = pi_vector.shape[1]
    e1 = np.ones(n)  # First column - all ones
    
    for attempt in range(max_attempts):
        # Generate a random matrix and check if it's invertible
        R = np.random.randn(n, n)
        
        # Skip if R is not invertible or problematic
        if np.linalg.matrix_rank(R) < n:
            continue
            
        # Compute dot products needed for the transformation
        Re1 = np.dot(R, e1)
        f1R = np.dot(pi_vector, R)
        f1Re1 = np.dot(pi_vector, Re1)
        
        # Check for numerical stability
        if abs(f1Re1) < 1e-10:
            continue
        
        # Compute U = R - (Re1⊗f1R)/(f1Re1)
        # This formula ensures U e1 = e1 and f1 U = [1, 0, 0, ..., 0]
        outer_product = np.outer(Re1, f1R)
        U = R - outer_product / f1Re1
        
        # Verify the requirements
        if (np.allclose(U[:, 0], 1, atol=1e-10) and  # First column is ones
            np.linalg.matrix_rank(U) == n and        # Full rank
            np.allclose(np.dot(pi_vector, U), np.array([1] + [0]*(n-1)), atol=1e-10)):  # First row of inverse
            return U
    
    # Fallback to the simpler method if all attempts fail
    return construct_U_simpler(pi_vector, max_attempts=1)

def generate_eigenvalues(second_largest_eigenvalue: float, U: np.ndarray) -> List[float]:
    """
    Generate a list of eigenvalues for the transition matrix A = U @ D @ U_inv, where D is a diagonal matrix 
    with the eigenvalues on the diagonal, largest is 1, ordered descending from 1 to 0,
    and A has all positive entries.
    
    This function is guaranteed to return valid eigenvalues that result in all positive entries for A.
    
    Parameters:
    second_largest_eigenvalue (float): Value of the second-largest eigenvalue (between 0 and 1)
    U (np.ndarray): The matrix U
    
    Returns:
    List[float]: List of eigenvalues for the diagonal matrix D
    """
    n = U.shape[0]
    
    # Check if second_largest_eigenvalue is valid
    if not (0 <= second_largest_eigenvalue < 1):
        raise ValueError("Second-largest eigenvalue must be between 0 and 1")
    
    # For n = 1, only one eigenvalue (1.0)
    if n == 1:
        return [1.0]
    
    # For n = 2, only two eigenvalues (1.0, second_largest)
    if n == 2:
        # Test if these eigenvalues result in positive A
        D = np.diag([1.0, second_largest_eigenvalue])
        U_inv = np.linalg.inv(U)
        A = U @ D @ U_inv
        
        if np.min(A) > 0:
            return [1.0, second_largest_eigenvalue]
        else:
            # If not, find the largest valid second eigenvalue
            lambda2 = binary_search_eigenvalue(U, 0, second_largest_eigenvalue)
            return [1.0, lambda2]
    
    # For n > 2, try optimization-based approach first
    try:
        eigenvalues = optimize_eigenvalues(second_largest_eigenvalue, U)
        
        # Verify the solution
        D = np.diag(eigenvalues)
        U_inv = np.linalg.inv(U)
        A = U @ D @ U_inv
        
        if np.min(A) > 0:
            return eigenvalues
    except Exception as e:
        print(f"Optimization failed: {e}. Trying alternative approaches.")
    
    # If optimization fails, try binary search approach
    return binary_search_approach(second_largest_eigenvalue, U)

def optimize_eigenvalues(second_largest_eigenvalue: float, U: np.ndarray) -> List[float]:
    """
    Use optimization to find eigenvalues that maximize the minimum element of A
    while ensuring A has all positive entries.
    """
    n = U.shape[0]
    U_inv = np.linalg.inv(U)
    
    # Initial guess: exponentially decreasing values
    x0 = np.array([second_largest_eigenvalue * (0.5 ** i) for i in range(n-2)])
    
    # Bounds for optimization: all eigenvalues between 0 and second_largest_eigenvalue
    bounds = [(1e-10, second_largest_eigenvalue) for _ in range(n-2)]
    
    def objective(x):
        # Construct eigenvalue diagonal matrix
        eigenvalues = np.concatenate(([1.0, second_largest_eigenvalue], x))
        D = np.diag(eigenvalues)
        
        # Compute transition matrix
        A = U @ D @ U_inv
        
        # Objective: negative of minimum element (we want to maximize the minimum element)
        min_element = np.min(A)
        
        # Penalty for negative elements
        penalty = 1000 * np.sum(np.maximum(0, -A))
        
        return -min_element + penalty
    
    # Run optimization
    result = minimize(
        objective, 
        x0, 
        bounds=bounds,
        method='L-BFGS-B', 
        options={'maxiter': 100, 'ftol': 1e-8}
    )
    
    # Check if optimization succeeded
    if not result.success:
        raise RuntimeError("Optimization failed to converge")
    
    # Construct final eigenvalues
    eigenvalues = np.concatenate(([1.0, second_largest_eigenvalue], result.x))
    
    return eigenvalues.tolist()

def binary_search_eigenvalue(U: np.ndarray, low: float, high: float, 
                            fixed_eigenvalues: List[float] = None) -> float:
    """
    Binary search to find the largest valid eigenvalue within a range that 
    results in a positive transition matrix.
    
    Parameters:
    U (np.ndarray): The matrix U
    low (float): Lower bound for the eigenvalue
    high (float): Upper bound for the eigenvalue
    fixed_eigenvalues (List[float]): List of eigenvalues that are already fixed
    
    Returns:
    float: The largest valid eigenvalue
    """
    n = U.shape[0]
    U_inv = np.linalg.inv(U)
    
    # If fixed_eigenvalues is None, initialize with just the first eigenvalue
    if fixed_eigenvalues is None:
        fixed_eigenvalues = [1.0]
    
    # Number of eigenvalues already fixed
    k = len(fixed_eigenvalues)
    
    # If we're searching for the last eigenvalue, we can be more precise
    tol = 1e-10 if k == n-1 else 1e-6
    max_iterations = 50
    
    best_valid_value = low  # Start with the known valid lower bound
    
    # Binary search
    for _ in range(max_iterations):
        if abs(high - low) < tol:
            break
            
        mid = (low + high) / 2
        
        # Create the full eigenvalue list
        eigenvalues = fixed_eigenvalues + [mid] + [0.0] * (n - k - 1)
        
        # Construct A
        D = np.diag(eigenvalues)
        A = U @ D @ U_inv
        
        # Check if all entries are positive
        if np.min(A) > 0:
            best_valid_value = mid
            low = mid  # Move up to find a larger valid value
        else:
            high = mid  # Move down to find a valid value
    
    return best_valid_value

def binary_search_approach(second_largest_eigenvalue: float, U: np.ndarray) -> List[float]:
    """
    Find valid eigenvalues using a hierarchical binary search approach.
    This approach is guaranteed to find valid eigenvalues.
    """
    n = U.shape[0]
    
    # Start with first eigenvalue fixed at 1.0
    eigenvalues = [1.0]
    
    # Find the valid second eigenvalue (no larger than given second_largest_eigenvalue)
    lambda2 = binary_search_eigenvalue(U, 0, second_largest_eigenvalue, eigenvalues)
    eigenvalues.append(lambda2)
    
    # Find remaining eigenvalues one by one
    for i in range(2, n):
        # The upper bound for the next eigenvalue is the last found eigenvalue
        lambda_i = binary_search_eigenvalue(U, 0, eigenvalues[-1], eigenvalues)
        eigenvalues.append(lambda_i)
    
    return eigenvalues

def theoretical_fallback(U: np.ndarray) -> List[float]:
    """
    Theoretical fallback that is guaranteed to produce a valid set of eigenvalues
    based on the properties of non-negative matrices.
    """
    n = U.shape[0]
    pi = np.linalg.inv(U)[0]  # First row of U⁻¹
    
    # Calculate minimum positive element in pi
    min_pi = np.min(pi[pi > 0])
    
    # Set eigenvalues to extremely small values except the first
    epsilon = min_pi / (100 * n)
    eigenvalues = [1.0] + [epsilon] * (n-1)
    
    return eigenvalues

def build_ergodic_unichain_transition_matrices(num_states: int) -> List[np.ndarray]:
    # define steady state
    pi_steady_state_lst = build_skew_static_distributions(num_states, 5)
    # check they are positive
    assert np.all(np.array(pi_steady_state_lst) > 0)
    
    # define second largest eigenvalue
    second_largest_eigenvalue = 0.9

    # build transition matrices
    transition_matrices = []
    for pi in pi_steady_state_lst:
        U = construct_U(pi)
        # U^-1 is the inverse of U
        U_inv = np.linalg.inv(U)
        # check the first row of U_inv is the steady state
        assert np.allclose(U_inv[0], pi)
        # D is a diagonal matrix with the eigenvalues on the diagonal, largest is 1, ordered descending from 1 to 0
        eigenvalues = generate_eigenvalues(second_largest_eigenvalue, U)
        D = np.diag(eigenvalues)
        # A is the transition matrix
        A = U @ D @ U_inv
        # check A is a stochastic matrix
        assert np.allclose(A.sum(axis=1), np.ones(num_states))
        # check A is ergodic
        assert np.all(np.linalg.matrix_power(A, num_states) > 0)

        # build transition matrices
        transition_matrices.append(A)
    
    return transition_matrices

def calculate_steady_state(matrix):
    """
    Calculate the steady state distribution of a stochastic matrix.
    
    For a stochastic matrix P, the steady state distribution π is a probability vector
    such that π = πP, i.e., π is a left eigenvector of P with eigenvalue 1.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        A stochastic matrix where each row sums to 1
        
    Returns:
    --------
    numpy.ndarray
        The steady state distribution vector
    """
    # Transpose the matrix for finding left eigenvectors
    matrix_t = matrix.T
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix_t)
    
    # Find index of eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    
    # Extract the corresponding eigenvector
    steady_state = np.real(eigenvectors[:, idx])
    
    # Normalize to ensure it sums to 1
    steady_state = steady_state / np.sum(steady_state)
    
    return steady_state

def calculate_entropy_rate(matrix, stationary_dist=None):
        """
        Calculate the entropy rate of a Markov chain with transition matrix P.
        
        For a Markov chain, the entropy rate is H(X) = sum_i μ_i * h_i
        where μ_i is the stationary distribution and
        h_i = -sum_j P_ij * log(P_ij) is the entropy of state i
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            A stochastic matrix (transition matrix) where each row sums to 1
        stationary_dist : numpy.ndarray, optional
            The stationary distribution of the chain. If None, it will be calculated.
            
        Returns:
        --------
        float
            The entropy rate of the Markov chain
        """
        if stationary_dist is None:
            stationary_dist = calculate_steady_state(matrix)
        
        # Calculate entropy for each state
        state_entropies = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            # Get transition probabilities from state i
            probs = matrix[i]
            # Calculate entropy for this state (row)
            mask = probs > 0
            state_entropies[i] = -np.sum(probs[mask] * np.log2(probs[mask]))
        
        # Calculate entropy rate using the formula H(X) = sum_i μ_i * h_i
        entropy_rate = np.sum(stationary_dist * state_entropies)
        
        return entropy_rate
    

if __name__ == "__main__":
    NUM_STATES = [4, 8, 16, 32, 64]
    NUM_OBSERVATIONS = [4, 8, 16, 32, 64]
    SEQ_LENGTH = [5, 10, 20, 50, 100]
    NUM_SEQUENCES = 100

    for num_states in NUM_STATES:
        for num_observations in NUM_OBSERVATIONS:
            for seq_length in SEQ_LENGTH:
                # A_list = build_transition_matrices(num_states, num_observations)
                A_list = build_ergodic_unichain_transition_matrices(num_states)
                print(A_list)
                entropy_rate_list = [calculate_entropy_rate(A) for A in A_list]
                print(entropy_rate_list)
                B_list = build_emission_matrices(num_states, num_observations)
                pis = build_initial_distribution(num_states)
                for A in A_list:
                    for B in B_list:
                        for pi in pis:
                            hmm = CustomHMM(np.arange(num_states), np.arange(num_observations), A, B, pi)
                            hmm.generate_dataset(NUM_SEQUENCES, seq_length, seed=5775709)

