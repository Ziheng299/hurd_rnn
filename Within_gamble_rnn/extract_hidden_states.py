import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import os
import pickle

from data_preprocessing import preprocess_pipeline
from model import create_model


def extract_hidden_states_from_model(model,
                                     dataloader,
                                     device,
                                     save_path: str = None) -> Dict:
    """
    Extract all hidden states from a trained RNN model.
    
    Returns:
        Dictionary with:
            - hidden_states: (N_sequences, 5_trials, hidden_dim)
            - subject_ids: (N_sequences,)
            - problem_ids: (N_sequences,)
            - choices: (N_sequences, 5)
            - contexts: (N_sequences, 5, 4)
            - problem_features: (N_sequences, 6)
    """
    model.eval()
    model = model.to(device)
    
    all_hidden_states = []
    all_subject_ids = []
    all_problem_ids = []
    all_choices = []
    all_contexts = []
    all_problem_features = []
    all_feedback_conditions = []
    
    print("Extracting hidden states...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx}/{len(dataloader)} batches...")
            
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            output = model(batch, return_hidden_states=True)
            
            hidden_states = output['hidden_states'].cpu().numpy()  # (B, 5, H)
            
            all_hidden_states.append(hidden_states)
            all_subject_ids.append(batch['subject_id'].cpu().numpy())
            all_problem_ids.append(batch['problem_id'].cpu().numpy())
            all_choices.append(batch['choices'].cpu().numpy())
            all_contexts.append(batch['contexts'].cpu().numpy())
            all_problem_features.append(batch['problem_features'].cpu().numpy())
            all_feedback_conditions.append(batch['feedback_condition'].cpu().numpy())
    
    # Concatenate all batches
    data = {
        'hidden_states': np.concatenate(all_hidden_states, axis=0),  # (N, 5, H)
        'subject_ids': np.concatenate(all_subject_ids, axis=0),       # (N,)
        'problem_ids': np.concatenate(all_problem_ids, axis=0),       # (N,)
        'choices': np.concatenate(all_choices, axis=0),               # (N, 5)
        'contexts': np.concatenate(all_contexts, axis=0),             # (N, 5, 4)
        'problem_features': np.concatenate(all_problem_features, axis=0),  # (N, 6)
        'feedback_conditions': np.concatenate(all_feedback_conditions, axis=0)  # (N, 1)
    }
    
    print(f"\nExtracted hidden states shape: {data['hidden_states'].shape}")
    print(f"Number of sequences: {len(data['subject_ids'])}")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        np.savez_compressed(save_path, **data)
        print(f"Saved hidden states to {save_path}")
        
        metadata_df = pd.DataFrame({
            'subject_id': data['subject_ids'],
            'problem_id': data['problem_ids'],
            'feedback_condition': data['feedback_conditions'].flatten()
        })
        csv_path = save_path.replace('.npz', '_metadata.csv')
        metadata_df.to_csv(csv_path, index=False)
        print(f"Saved metadata to {csv_path}")
    
    return data


def load_hidden_states(path: str) -> Dict:
    """Load previously saved hidden states."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


def extract_from_saved_model(model_path: str,
                             data_path_obj: str,
                             data_path_subj: str,
                             model_kwargs: Dict,
                             output_path: str,
                             device: torch.device,
                             batch_size: int = 64):
    """
    Load a saved model and extract hidden states from all data.
    
    Args:
        model_path: Path to saved model (.pt file)
        data_path_obj: Path to objective features CSV
        data_path_subj: Path to subject data CSV
        model_kwargs: Arguments for model creation
        output_path: Where to save extracted hidden states
        device: Device to use
        batch_size: Batch size for extraction
    """
    print("="*80)
    print("EXTRACTING HIDDEN STATES FROM SAVED MODEL")
    print("="*80)
    
    data = preprocess_pipeline(
        obj_path=data_path_obj,
        subj_path=data_path_subj,
        test_size=0.2,
        random_state=42,
        batch_size=batch_size
    )
    
    all_sequences = data['train_sequences'] + data['test_sequences']
    
    from data_preprocessing import DecisionMakingDataset, DataLoader
    full_dataset = DecisionMakingDataset(all_sequences)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = create_model(model_type='recurrent', **model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {model_path}")
    
    # Extract hidden states
    print("\nExtracting hidden states...")
    hidden_data = extract_hidden_states_from_model(
        model=model,
        dataloader=full_loader,
        device=device,
        save_path=output_path
    )
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Hidden states saved to: {output_path}")
    print(f"Shape: {hidden_data['hidden_states'].shape}")
    
    return hidden_data