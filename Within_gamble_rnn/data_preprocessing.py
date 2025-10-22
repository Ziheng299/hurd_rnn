import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, List, Tuple


class DecisionMakingDataset(Dataset):
    """
    PyTorch Dataset for decision-making sequences.
    Each sample is a complete 5-trial sequence for one subject–problem pair.
    """
    def __init__(self, sequences: List[Dict]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "subject_id": seq["subject_id"],
            "problem_id": seq["problem_id"],
            "problem_features": torch.FloatTensor(seq["problem_features"]),  # [Ha, La, pHa, Hb, Lb, pHb]
            "feedback_condition": torch.FloatTensor([seq["feedback_condition"]]),  # 0 or 1
            "contexts": torch.FloatTensor(seq["contexts"]),  # (5, 4): [choice, reward, forgone, flag]
            "choices": torch.LongTensor(seq["choices"]),     # (5,): 0=A, 1=B
        }


def load_and_merge_data(obj_path: str, subj_path: str) -> pd.DataFrame:
    """
    Load and merge objective features and subject data.
    Assumes consistent 'uniqueID' key. Falls back to 'Problem' if needed.
    """
    obj_df = pd.read_csv(obj_path)
    subj_df = pd.read_csv(subj_path, low_memory=False)

    merge_key = "uniqueID" if ("uniqueID" in obj_df.columns and "uniqueID" in subj_df.columns) else "Problem"
    merged_df = pd.merge(subj_df, obj_df, on=merge_key)

    if "Feedback_x" in merged_df.columns and "Feedback_y" in merged_df.columns:
        merged_df = merged_df.drop(columns=["Feedback_y"]).rename(columns={"Feedback_x": "Feedback"})

    print("\nMerged shape:", merged_df.shape)
    print("Sample merged columns:", merged_df.columns.tolist())
    return merged_df


def prepare_context(choice: int, reward: float, forgone: float, feedback_condition: int) -> List[float]:
    """
    Prepare context vector [choice, reward, forgone, flag] for RNN input.
    """
    if feedback_condition == 1:
        return [float(choice), float(reward), float(forgone), 1.0]
    else:
        return [float(choice), 0.0, 0.0, 0.0]


def create_sequences(merged_df: pd.DataFrame) -> List[Dict]:
    """
    Convert merged dataframe into a list of 5-trial sequences.
    """
    sequences: List[Dict] = []
    skipped = 0

    for _, row in merged_df.iterrows():
        problem_features = [
            row["Ha"], row["La"], row["pHa"],
            row["Hb"], row["Lb"], row["pHb"]
        ]

        feedback = 1 if (row["Feedback"] is True or str(row["Feedback"]).upper().strip() == "TRUE") else 0

        has_all_trials = all(pd.notna(row.get(f"sel{t}", np.nan)) for t in range(1, 6))
        if not has_all_trials:
            skipped += 1
            continue

        choices: List[int] = []
        contexts: List[List[float]] = []
        for t in range(1, 6):
            choice_str = str(row[f"sel{t}"]).strip().upper()
            choice = 0 if choice_str == "A" else 1
            choices.append(choice)

            if t == 1:
                context = [0.0, 0.0, 0.0, float(feedback)]
            else:
                prev_choice_str = str(row[f"sel{t-1}"]).strip().upper()
                prev_choice = 0 if prev_choice_str == "A" else 1
                prev_reward = row[f"reward{t-1}"]
                prev_forgone = row[f"forgone{t-1}"]

                context = prepare_context(prev_choice, prev_reward, prev_forgone, feedback)

            contexts.append(context)

        sequence = {
            "subject_id": row["subjectID"],
            "problem_id": row["uniqueID"] if "uniqueID" in row else row["Problem"],
            "problem_features": problem_features,
            "feedback_condition": feedback,
            "contexts": contexts,  # shape (5, 4)
            "choices": choices,    # shape (5,)
        }
        sequences.append(sequence)

    print(f"Created {len(sequences)} sequences")
    if skipped > 0:
        print(f"Skipped {skipped} sequences due to missing data")
    return sequences


def train_test_split_by_subject_problem(sequences: List[Dict], 
                                        test_size: float = 0.2,
                                        random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split sequences by subject*problem pairs.
    Each sequence = ONE subject*problem pair with ALL 5 trials.
    """
    np.random.seed(random_state)
    
    n_total = len(sequences)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_sequences = [sequences[i] for i in train_idx]
    test_sequences = [sequences[i] for i in test_idx]
    
    print(f"Total subject*problem pairs: {n_total}")
    print(f"Train pairs: {len(train_sequences)} ({len(train_sequences)/n_total*100:.1f}%)")
    print(f"Test pairs: {len(test_sequences)} ({len(test_sequences)/n_total*100:.1f}%)")
    
    return train_sequences, test_sequences


def sample_train_data(train_sequences: List[Dict],
                     percentage: float,
                     random_state: int = 42) -> List[Dict]:
    """
    Sample a percentage of training sequences for learning curve experiments.
    """
    np.random.seed(random_state)
    
    n_samples = max(1, int(len(train_sequences) * percentage / 100.0))
    sampled_idx = np.random.choice(len(train_sequences), size=n_samples, replace=False)
    
    return [train_sequences[i] for i in sampled_idx]


def get_dataloaders(train_sequences: List[Dict],
                   test_sequences: List[Dict],
                   batch_size: int = 64,
                   num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders.
    """
    train_dataset = DecisionMakingDataset(train_sequences)
    test_dataset = DecisionMakingDataset(test_sequences)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def preprocess_pipeline(obj_path: str,
                       subj_path: str,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       batch_size: int = 64) -> Dict:
    """
    Complete preprocessing pipeline.
    
    Returns:
        Dictionary with train_loader, test_loader, train_sequences, test_sequences
    """
    merged_df = load_and_merge_data(obj_path, subj_path)
    sequences = create_sequences(merged_df)
    
    train_sequences, test_sequences = train_test_split_by_subject_problem(
        sequences, test_size=test_size, random_state=random_state
    )
    
    train_loader, test_loader = get_dataloaders(
        train_sequences, test_sequences, batch_size=batch_size
    )
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_sequences': train_sequences,
        'test_sequences': test_sequences,
    }