import torch

# Data Paths 
OBJ_PATH = "/Users/ziheng/Desktop/NYU/RNN4Cognitive/hurd_rnn/Data/c13k_obj_feats_uid.csv"
SUBJ_PATH = "/Users/ziheng/Desktop/NYU/RNN4Cognitive/hurd_rnn/Data/c13k_subject_data_uid.csv"
OUTPUT_DIR = "./results"

# Baseline settings
UTILITY_HIDDEN_DIM = 32

# RNN settings
HIDDEN_STATE_DIM = 16
RNN_TYPE = "gru"  # "gru" or "lstm"
USE_SUBJECT_H0 = False  
STOCHASTIC_SPEC = "softmax"  # "softmax" or "constant-error"

# Training Settings
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Regularization
USE_REGULARIZATION = True
LAMBDA_L2 = 1e-4

# Data Split Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Learning Curve Settings 
TRAIN_PERCENTAGES = [0.1, 0.3, 0.7, 1, 3, 10, 30, 50, 70, 100]

# Device Settings
# "auto", "cpu", or "cuda"
DEVICE = "auto"
VERBOSE = True
def get_device():
    """Get torch device based on config."""
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)