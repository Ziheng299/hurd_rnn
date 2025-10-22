# model.py

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from neural_eu import BaselineUtilityNetwork, RecurrentUtilityNetwork, UtilityEvaluator, StochasticChoice
from first_layer_rnn import RNNCell, SubjectSpecificHiddenStates


class BaselineEUModel(nn.Module):
    """
    Baseline Expected Utility model 
    """
    def __init__(self, 
                 utility_hidden_dim: int = 32,
                 stochastic_spec: str = "softmax"):
        super(BaselineEUModel, self).__init__()
        
        self.utility_net = BaselineUtilityNetwork(hidden_dim=utility_hidden_dim)
        self.utility_evaluator = UtilityEvaluator(self.utility_net)
        self.stochastic_layer = StochasticChoice(stochastic_spec=stochastic_spec)
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass for baseline model.
        Returns:
            Dictionary with:
                - predictions: (batch_size, 5) - P(choose A) for each trial
                - loss_info: dict with per-trial info if needed
        """
        problem_features = batch['problem_features']
        batch_size = problem_features.shape[0]
        
        utilities_dict = self.utility_evaluator.evaluate_gamble_utilities(problem_features)
        V_A, V_B = self.utility_evaluator.compute_expected_utilities(utilities_dict)
        
        P_A = self.stochastic_layer(V_A, V_B)  # (batch_size,)
        predictions = P_A.unsqueeze(1).expand(-1, 5)  # (batch_size, 5)
        
        return {
            'predictions': predictions,  # (batch_size, 5)
            'V_A': V_A,
            'V_B': V_B
        }


class RecurrentEUModel(nn.Module):
    """
    Recurrent Expected Utility model (with RNN).
    Utility function u(x|h[t]) changes over trials based on hidden state.
    """
    def __init__(self,
                 hidden_state_dim: int = 16,
                 utility_hidden_dim: int = 32,
                 rnn_type: str = "gru",
                 stochastic_spec: str = "softmax",
                 use_subject_h0: bool = False,
                 num_subjects: Optional[int] = None):
        super(RecurrentEUModel, self).__init__()
        
        self.hidden_state_dim = hidden_state_dim
        self.rnn_type = rnn_type
        self.use_subject_h0 = use_subject_h0
        
        self.rnn_cell = RNNCell(hidden_state_dim=hidden_state_dim, rnn_type=rnn_type)
        
        # Subject-specific h0 
        if use_subject_h0:
            if num_subjects is None:
                raise ValueError("num_subjects required when use_subject_h0=True")
            self.subject_h0 = SubjectSpecificHiddenStates(
                num_subjects=num_subjects,
                hidden_state_dim=hidden_state_dim,
                rnn_type=rnn_type
            )
        
        self.utility_net = RecurrentUtilityNetwork(
            hidden_state_dim=hidden_state_dim,
            utility_hidden_dim=utility_hidden_dim
        )
        self.utility_evaluator = UtilityEvaluator(self.utility_net)
        self.stochastic_layer = StochasticChoice(stochastic_spec=stochastic_spec)
    
    def get_initial_hidden(self, 
                          batch_size: int, 
                          subject_ids: Optional[torch.Tensor], 
                          device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get initial hidden state h[0] (and c[0] for LSTM).
        """
        if self.use_subject_h0:
            if subject_ids is None:
                raise ValueError("subject_ids required when use_subject_h0=True")
            h0, c0 = self.subject_h0(subject_ids)
        else:
            h0, c0 = self.rnn_cell.init_hidden(batch_size, device)
        
        return h0, c0
    
    def forward(self, batch: Dict, return_hidden_states: bool = False) -> Dict:
        """
        Forward pass for recurrent model over 5 trials.

        Returns:
            Dictionary with:
                - predictions: (batch_size, 5) - P(choose A) for each trial
                - hidden_states: (batch_size, 5, hidden_dim) - if return_hidden_states=True
                - V_A_all: (batch_size, 5) - expected utilities
                - V_B_all: (batch_size, 5)
        """
        problem_features = batch['problem_features']  # (batch_size, 6)
        contexts = batch['contexts']  # (batch_size, 5, 4)
        batch_size = problem_features.shape[0]
        device = problem_features.device
        
        subject_ids = batch.get('subject_id', None)
        h, c = self.get_initial_hidden(batch_size, subject_ids, device)
        
        predictions = []
        V_A_list = []
        V_B_list = []
        hidden_states = [] if return_hidden_states else None
        
        for t in range(5):
            if return_hidden_states:
                hidden_states.append(h.detach().clone())
            
            utilities_dict = self.utility_evaluator.evaluate_gamble_utilities(
                problem_features, h=h
            )
            V_A, V_B = self.utility_evaluator.compute_expected_utilities(utilities_dict)
            P_A = self.stochastic_layer(V_A, V_B)  # (batch_size,)
            
            predictions.append(P_A)
            V_A_list.append(V_A)
            V_B_list.append(V_B)
            
            context_t = contexts[:, t, :]  # (batch_size, 4)
            h, c = self.rnn_cell(context_t, h, c)
    
        predictions = torch.stack(predictions, dim=1)
        V_A_all = torch.stack(V_A_list, dim=1)
        V_B_all = torch.stack(V_B_list, dim=1)
        
        result = {
            'predictions': predictions,
            'V_A_all': V_A_all,
            'V_B_all': V_B_all
        }
        
        if return_hidden_states:
            result['hidden_states'] = torch.stack(hidden_states, dim=1)
        
        return result
    
    def predict_single_trial(self, 
                            problem_features: torch.Tensor,
                            h: torch.Tensor,
                            c: Optional[torch.Tensor] = None) -> Dict:
        utilities_dict = self.utility_evaluator.evaluate_gamble_utilities(
            problem_features, h=h
        )
        V_A, V_B = self.utility_evaluator.compute_expected_utilities(utilities_dict)
        P_A = self.stochastic_layer(V_A, V_B)
        
        return {
            'P_A': P_A,
            'V_A': V_A,
            'V_B': V_B,
            'utilities': utilities_dict
        }


def create_model(model_type: str = "baseline",
                hidden_state_dim: int = 16,
                utility_hidden_dim: int = 32,
                rnn_type: str = "gru",
                stochastic_spec: str = "softmax",
                use_subject_h0: bool = False,
                num_subjects: Optional[int] = None) -> nn.Module:
    if model_type == "baseline":
        return BaselineEUModel(
            utility_hidden_dim=utility_hidden_dim,
            stochastic_spec=stochastic_spec
        )
    elif model_type == "recurrent":
        return RecurrentEUModel(
            hidden_state_dim=hidden_state_dim,
            utility_hidden_dim=utility_hidden_dim,
            rnn_type=rnn_type,
            stochastic_spec=stochastic_spec,
            use_subject_h0=use_subject_h0,
            num_subjects=num_subjects
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")