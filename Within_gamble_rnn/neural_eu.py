import torch
import torch.nn as nn
from typing import Optional

class BaselineUtilityNetwork(nn.Module):
    """
    Baseline utility network.
    Architecture: x → Linear → Sigmoid → Linear → u(x)
    """
    def __init__(self, hidden_dim: int = 32):
        super(BaselineUtilityNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),    
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1)  
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute utility for outcome values.
        
        Returns:
            u(x): Utilities, shape (batch_size, 1)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        utility = self.network(x)
        return utility


class RecurrentUtilityNetwork(nn.Module):
    """
    Recurrent utility network (with hidden state).
    Maps outcome values and hidden states to utilities: u(x|h)

    Architecture: [x, h] → Linear → Sigmoid → Linear → u(x|h)
    """
    def __init__(self, hidden_state_dim: int, utility_hidden_dim: int = 32):
        super(RecurrentUtilityNetwork, self).__init__()
        
        self.hidden_state_dim = hidden_state_dim
        
        # Input: [x, h] concatenated
        input_dim = 1 + hidden_state_dim  # 1 for outcome x, rest for h
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, utility_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(utility_hidden_dim, 1)  
        )
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute utility for outcome values conditioned on hidden state.
        
        Returns:
            u(x|h): Utilities, shape (batch_size, 1)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        x_h = torch.cat([x, h], dim=-1) 
        utility = self.network(x_h)
        return utility


class UtilityEvaluator:
    """
    Helper class to evaluate utilities for all outcomes in a gamble pair.
    Works with both baseline and recurrent utility networks.
    """
    def __init__(self, utility_network: nn.Module):
        """
        Args:
            utility_network: Either BaselineUtilityNetwork or RecurrentUtilityNetwork
        """
        self.utility_network = utility_network
        self.is_recurrent = isinstance(utility_network, RecurrentUtilityNetwork)
    
    def evaluate_gamble_utilities(self,
                                  problem_features: torch.Tensor,
                                  h: Optional[torch.Tensor] = None) -> dict:
        """
        Evaluate utilities for all 4 outcomes in a gamble pair.
        
        Returns:
            Dictionary with utilities and probabilities
        """
        batch_size = problem_features.shape[0]
        
        Ha = problem_features[:, 0]  
        La = problem_features[:, 1]
        pHa = problem_features[:, 2]
        Hb = problem_features[:, 3]
        Lb = problem_features[:, 4]
        pHb = problem_features[:, 5]
        
        # Compute utilities for all 4 outcomes
        if self.is_recurrent:
            if h is None:
                raise ValueError("Hidden state h is required for RecurrentUtilityNetwork")
            u_Ha = self.utility_network(Ha, h) 
            u_La = self.utility_network(La, h)
            u_Hb = self.utility_network(Hb, h)
            u_Lb = self.utility_network(Lb, h)
        else:
            u_Ha = self.utility_network(Ha)
            u_La = self.utility_network(La)
            u_Hb = self.utility_network(Hb)
            u_Lb = self.utility_network(Lb)
        
        u_Ha = u_Ha.squeeze(-1)
        u_La = u_La.squeeze(-1)
        u_Hb = u_Hb.squeeze(-1)
        u_Lb = u_Lb.squeeze(-1)
        
        return {
            'u_Ha': u_Ha,
            'u_La': u_La,
            'u_Hb': u_Hb,
            'u_Lb': u_Lb,
            'pHa': pHa,
            'pLa': 1.0 - pHa, 
            'pHb': pHb,
            'pLb': 1.0 - pHb
        }
    
    def compute_expected_utilities(self, utilities_dict: dict) -> tuple:
        """
        Compute expected utilities V(A) and V(B).
        
        Returns:
            (V_A, V_B): Expected utilities, each shape (batch_size,)
        """
        V_A = (utilities_dict['u_Ha'] * utilities_dict['pHa'] + 
               utilities_dict['u_La'] * utilities_dict['pLa'])
    
        V_B = (utilities_dict['u_Hb'] * utilities_dict['pHb'] + 
               utilities_dict['u_Lb'] * utilities_dict['pLb'])
        
        return V_A, V_B


class StochasticChoice(nn.Module):
    """
    Stochastic choice layer that converts expected utilities to choice probabilities.
    """
    def __init__(self, stochastic_spec: str = "softmax"):
        super(StochasticChoice, self).__init__()
        
        self.stochastic_spec = stochastic_spec
        
        # Temperature parameter (log-scale for positivity)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self._tau_eps = 1e-4
        
        if stochastic_spec == "constant-error":
            # Error parameter for constant-error spec
            self.mu = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, V_A: torch.Tensor, V_B: torch.Tensor) -> torch.Tensor:
        """
        Compute probability of choosing gamble A.
        """
        if self.stochastic_spec == "softmax":
            return self._softmax_choice(V_A, V_B)
        else:  # constant-error
            return self._constant_error_choice(V_A, V_B)
    
    def _softmax_choice(self, V_A: torch.Tensor, V_B: torch.Tensor) -> torch.Tensor:
        """
        Softmax choice: P(A) = exp(T*V_A) / [exp(T*V_A) + exp(T*V_B)]
        """
        T = torch.nn.functional.softplus(self.log_tau) + self._tau_eps
        logits = torch.stack([T * V_A, T * V_B], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 0]
    
    def _constant_error_choice(self, V_A: torch.Tensor, V_B: torch.Tensor) -> torch.Tensor:
        """
        Constant-error choice: Binary decision with error rate.
        From He, Zhao, & Bhatia (2020).
        
        If V_A >= V_B: P(A) = 1 - error, P(B) = error
        If V_A < V_B:  P(A) = error,     P(B) = 1 - error
        where error = sigmoid(mu) / 2
        """
        error = torch.sigmoid(self.mu) / 2.0
        choose_A = (V_A >= V_B).float()
        P_A = choose_A * (1.0 - error) + (1.0 - choose_A) * error
        
        return P_A