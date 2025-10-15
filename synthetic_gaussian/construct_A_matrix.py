import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from scipy.stats import beta
import os

class MatrixParametrizer(nn.Module):
    """
    Neural network to parametrize matrices U and U^-1 such that:
    1. First column of U is all ones
    2. First row of U^-1 is fixed (given)
    3. A = UDU^-1 has positive elements
    4. Each row of A sums to 1
    """
    def __init__(self, n, w, lambda2, device='cpu'):
        """
        Initialize the parametrization.
        
        Args:
            n: The dimension of the matrices (n x n)
            w: The first row of U^-1 (must be provided)
            lambda2: The second largest eigenvalue
            device: The device to use (cpu or cuda)
        """
        super().__init__()
        self.n = n
        self.device = device
        self.register_buffer('w', torch.tensor(w, dtype=torch.float32, device=device))
        self.register_buffer('lambda12', torch.tensor([1, lambda2], dtype=torch.float32, device=device))
        self.lambda2 = lambda2
        
        # Create parameters for U (excluding first column)
        self.U = nn.Parameter(torch.randn(n, n-1, device=device))

        # Create parameters for sigma
        self.Sigma = nn.Parameter(torch.randn(n-2, device=device))
        
    def get_U(self):
        """Construct U from the parametrization"""

        # First column is all ones
        u1 = torch.ones(self.n, 1, device=self.device)

        # Combine to form U
        U = torch.cat([u1, self.U], dim=1)
        return U
    
    def get_U_inv(self):
        """Construct U^-1 such that U^-1 @ U = I and the first row is w"""
        U = self.get_U()
        
        # First row is w
        U_inv_1 = self.w.view(1, -1)
        
        # compute pseudo inverse of U
        U_inv_rest = torch.pinverse(U)[1:]

        # Combine all rows
        U_inv = torch.cat([U_inv_1, U_inv_rest], dim=0)
        
        return U_inv
    
    def get_Sigma(self):
        return torch.diag(torch.cat([self.lambda12, self.Sigma]))
    
    def get_A(self):
        """Compute A = UDU^-1"""
        U = self.get_U()
        U_inv = self.get_U_inv()
        Sigma = self.get_Sigma()
        A = torch.matmul(torch.matmul(U, Sigma), U_inv)
        return A
    
    def forward(self):
        """
        Compute the loss for optimization.
        
        Returns:
            Dictionary of losses and the matrix A
        """
        # Compute A = UDU^-1
        A = self.get_A()
        
        # 1. Positivity loss: all elements of A must be positive
        positivity_loss = torch.sum(torch.clamp(-A, min=0))
        
        # 2. Row sum loss: each row of A must sum to 1
        row_sums = torch.sum(A, dim=1)
        row_sum_loss = torch.sum((row_sums - 1.0) ** 2)
        
        # 3. Orthogonality loss: U^-1 @ U = I
        U = self.get_U()
        U_inv = self.get_U_inv()
        I = torch.eye(self.n, device=self.device)
        orthogonality_loss = torch.sum((torch.matmul(U_inv, U) - I) ** 2)

        # all other eigenvalues are less than lambda2
        eigenvalue_loss = torch.sum(torch.clamp(torch.abs(self.Sigma) - self.lambda2, min=0))
        
        # Total loss
        total_loss = positivity_loss + row_sum_loss + orthogonality_loss + eigenvalue_loss
        
        return {
            'total_loss': total_loss,
            'positivity_loss': positivity_loss,
            'row_sum_loss': row_sum_loss,
            'orthogonality_loss': orthogonality_loss,
            'eigenvalue_loss': eigenvalue_loss,
            'A': A
        }

def train_matrix_parametrizer(n, w, lambda_2, num_epochs=5000, 
                              learning_rate=0.01, device='cpu', verbose=True):
    """
    Train the matrix parametrizer to find U and U^-1.
    
    Args:
        n: The dimension of the matrices (n x n)
        w: The first row of U^-1
        lambda_2: second largest eigen value
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: The device to use (cpu or cuda)
        verbose: Whether to print progress
        
    Returns:
        The trained model, loss history, and the final matrices U, U^-1, and A
    """
    # Initialize model
    model = MatrixParametrizer(n, w, lambda_2, device)
    model.to(device)
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    loss_history = []
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(num_epochs):
        # Forward pass
        result = model()
        loss = result['total_loss']
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss
        loss_history.append(loss.item())
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()
        
        # Print progress
        if verbose and (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print(f'  Positivity Loss: {result["positivity_loss"].item():.4f}')
            print(f'  Row Sum Loss: {result["row_sum_loss"].item():.4f}')
            print(f'  Orthogonality Loss: {result["orthogonality_loss"].item():.4f}')
            print(f'  Eigenvalue Loss: {result["eigenvalue_loss"].item():.4f}')
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Get final matrices
    U = model.get_U()
    U_inv = model.get_U_inv()
    A = model.get_A()
    Sigma = model.get_Sigma()
    
    return model, loss_history, U, U_inv, A, Sigma

if __name__ == "__main__":

    NUM_STATES = [4, 8, 16, 32, 64]
    STEADY_STATE_DISTRIBUTION = [1, 4, 16]
    MIXING_RATE = [0.99, 0.95, 0.75, 0.5]

    REPEATS = 100
    MAX_NUM_RETRY = 100

    As = []
    all_pi_0, all_lambda2 = [], []
    Us, U_invs = [], []
    Sigmas = []
    num_state = []
    entropies = []

    # ==========================================================================
    # for different number of states
    for num_states in NUM_STATES:

        pi = []
        for alpha in STEADY_STATE_DISTRIBUTION:
            dist = beta.pdf(np.linspace(0, 1, num_states + 1), 1, alpha)[:-1]
            pi.append(dist / np.sum(dist))
        
        # for different steady state distribution
        for pi_0 in pi:

            # for different mixing rates
            for lambda2 in MIXING_RATE:

                num_valid = 0
                failures = 0
                print('=' * 50)
                print(f"num_states: {num_states}")
                print(f"pi_0: {pi_0}")
                print(f"lambda2: {lambda2}")

                while(num_valid < REPEATS and failures < MAX_NUM_RETRY):

                    # initial attempt
                    model, loss_history, U, U_inv, A, Sigma = train_matrix_parametrizer(num_states, pi_0, lambda2, num_epochs=5000, learning_rate=0.01, device='cuda', verbose=False)
                    A = A.detach().cpu().numpy()
                    U = U.detach().cpu().numpy()
                    U_inv = U_inv.detach().cpu().numpy()
                    Sigma = Sigma.detach().cpu().numpy()
                    failures += 1

                    # print(f"A: {A}")
                    # print(f"U: {U}")
                    # print(f"U_inv: {U_inv}")
                    # print(f"Sigma: {Sigma}")

                    # check if U, U_inv are valid
                    if not np.all(np.isclose(U @ U_inv, np.eye(num_states), atol=1e-2)):
                        print("U @ U_inv != I")
                        continue

                    # check first column of U are all 1s
                    if not np.all(np.isclose(U[:, 0], np.ones(num_states))):
                        print("U[:, 0] != 1")
                        continue

                    # check first row of U_inv are all pi_0
                    if not np.all(np.isclose(U_inv[0, :], pi_0)):
                        print("U_inv[0, :] != pi_0")
                        continue

                    # check sigma
                    if Sigma[0, 0] != 1 or Sigma[1, 1] != lambda2 or (not np.all(Sigma.diagonal()[2:] < lambda2)):
                        print("sigma is incorrect")
                        continue

                    # check A
                    if not np.all(A > 0):
                        print("A contains negative entries")
                        continue
                    if not np.all(np.isclose(np.sum(A, 1), np.ones(num_states))):
                        print("A does not sum to 1")
                        continue

                    # save
                    failures = 0
                    all_pi_0.append(pi_0)
                    all_lambda2.append(lambda2)
                    num_state.append(num_states)
                    As.append(A)
                    Us.append(U)
                    U_invs.append(U_inv)
                    Sigmas.append(Sigma)
                    entropies.append((np.array(pi_0) * (-A * np.log2(A)).sum(1)).sum().item())
                    num_valid += 1
                    print(f"num_valid: {num_valid} | entropy: {(np.array(pi_0) * (-A * np.log2(A)).sum(1)).sum().item()}")

    DATA_PATH = 'data/'
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, 'A_matrix.pickle'), 'wb') as f:
        pickle.dump((num_state, all_pi_0, all_lambda2, Us, Sigmas, U_invs, As, entropies), f)
