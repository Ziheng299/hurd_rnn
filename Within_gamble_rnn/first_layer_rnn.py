import torch
import torch.nn as nn

class RNNCell(nn.Module):
    """
    RNN cell that updates hidden state based on context (choice, reward, forgone, feedback flag).
    
    h[t] = RNN(h[t-1], context[t])
    where context[t] = [choice_{t-1}, reward_{t-1}, forgone_{t-1}, feedback_flag]
    """
    def __init__(self, hidden_state_dim: int = 16, rnn_type: str = "gru"):
        super(RNNCell, self).__init__()
        
        self.hidden_state_dim = hidden_state_dim
        self.rnn_type = rnn_type
        
        context_dim = 4  # [choice, reward, forgone, flag]
        
        if rnn_type == "lstm":
            self.rnn = nn.LSTMCell(input_size=context_dim, hidden_size=hidden_state_dim)
        elif rnn_type == "gru":
            self.rnn = nn.GRUCell(input_size=context_dim, hidden_size=hidden_state_dim)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
    
    def forward(self, context: torch.Tensor, h_prev: torch.Tensor, 
                c_prev: torch.Tensor = None) -> tuple:
        """
        Update hidden state based on context.
        
        Args:
            context: Context vector [choice, reward, forgone, flag], shape (batch_size, 4)
            h_prev: Previous hidden state, shape (batch_size, hidden_state_dim)
            c_prev: Previous cell state (only for LSTM), shape (batch_size, hidden_state_dim)
        
        Returns:
            If LSTM: (h_new, c_new)
            If GRU: (h_new, None)
        """
        if self.rnn_type == "lstm":
            if c_prev is None:
                raise ValueError("c_prev required for LSTM")
            h_new, c_new = self.rnn(context, (h_prev, c_prev))
            return h_new, c_new
        else:
            h_new = self.rnn(context, h_prev)
            return h_new, None
    
    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """
        Initialize hidden state (and cell state for LSTM).
        """
        h0 = torch.zeros(batch_size, self.hidden_state_dim, device=device)
        if self.rnn_type == "lstm":
            c0 = torch.zeros(batch_size, self.hidden_state_dim, device=device)
            return h0, c0
        else:
            return h0, None


class SubjectSpecificHiddenStates(nn.Module):
    """
    Learnable initial hidden states for each subject.
    Maps subject IDs to their personalized h0.
    """
    def __init__(self, num_subjects: int, hidden_state_dim: int, rnn_type: str = "gru"):
        super(SubjectSpecificHiddenStates, self).__init__()
        
        self.num_subjects = num_subjects
        self.hidden_state_dim = hidden_state_dim
        self.rnn_type = rnn_type
        
        self.h0_embedding = nn.Embedding(num_subjects, hidden_state_dim)
        nn.init.normal_(self.h0_embedding.weight, mean=0.0, std=0.01)
        
        if rnn_type == "lstm":
            self.c0_embedding = nn.Embedding(num_subjects, hidden_state_dim)
            nn.init.normal_(self.c0_embedding.weight, mean=0.0, std=0.01)
    
    def forward(self, subject_ids: torch.Tensor) -> tuple:
        """
        Get initial hidden states for given subjects.
        """
        subject_ids = subject_ids.long()
        h0 = self.h0_embedding(subject_ids)
        
        if self.rnn_type == "lstm":
            c0 = self.c0_embedding(subject_ids)
            return h0, c0
        else:
            return h0, None
        
    def get_population_mean(self) -> tuple:
        """
        Get mean h0 across all subjects (useful for unknown subjects).
        """
        h0_mean = self.h0_embedding.weight.mean(dim=0)
        
        if self.rnn_type == "lstm":
            c0_mean = self.c0_embedding.weight.mean(dim=0)
            return h0_mean, c0_mean
        else:
            return h0_mean, None
        
    def regularization_loss(self, lambda_l2: float = 1e-4) -> torch.Tensor:
        reg = lambda_l2 * (self.h0_embedding.weight ** 2).mean()
        if self.rnn_type == "lstm":
            reg = 0.5 * (reg + lambda_l2 * (self.c0_embedding.weight ** 2).mean())
        return reg
