import torch
import argparse
import json
import os
from datetime import datetime

from data_preprocessing import preprocess_pipeline
from model import create_model
from extract_hidden_states import extract_from_saved_model
from analyze_hidden_states import complete_analysis_pipeline
import config


def parse_args():
    """
    Parse command line arguments for interpretation.
    """
    parser = argparse.ArgumentParser(description='Interpret and Analyze Trained RNN Models')
    
    # paths to trained model and data
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--obj_path', type=str, default=config.OBJ_PATH,
                       help='Path to c13k_obj_feats_uid.csv')
    parser.add_argument('--subj_path', type=str, default=config.SUBJ_PATH,
                       help='Path to c13k_subject_data_uid.csv')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./interpretation_results',
                       help='Directory to save interpretation results')
    
    # Model architecture 
    parser.add_argument('--hidden_state_dim', type=int, default=config.HIDDEN_STATE_DIM,
                       help='Hidden state dimension (must match trained model)')
    parser.add_argument('--utility_hidden_dim', type=int, default=config.UTILITY_HIDDEN_DIM,
                       help='Utility network hidden dim (must match trained model)')
    parser.add_argument('--rnn_type', type=str, default=config.RNN_TYPE,
                       choices=['lstm', 'gru'],
                       help='RNN type (must match trained model)')
    parser.add_argument('--use_subject_h0', action='store_true',
                       default=config.USE_SUBJECT_H0,
                       help='Whether model uses subject-specific h0')
    
    # Analysis settings
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters (None = use elbow method)')
    parser.add_argument('--skip_extraction', action='store_true',
                       help='Skip extraction if hidden_states.npz already exists')
    parser.add_argument('--skip_analysis', action='store_true',
                       help='Only extract hidden states, skip analysis')
    
    # Data settings
    parser.add_argument('--test_size', type=float, default=config.TEST_SIZE,
                       help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=config.RANDOM_STATE,
                       help='Random seed')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for extraction')
    
    # Device
    parser.add_argument('--device', type=str, default=config.DEVICE,
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return device


def get_num_subjects(data_path: str) -> int:
    """Get number of unique subjects from data."""
    import pandas as pd
    df = pd.read_csv(data_path)
    return df['uid'].nunique()


def main():
    """
    Main interpretation pipeline.
    """
    args = parse_args()
    
    device = setup_device(args.device)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, 'interpretation_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to {config_path}")
    
    print("\n" + "="*80)
    print("RECURRENT NEURAL EU - INTERPRETATION PIPELINE")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Output: {output_dir}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Get number of subjects if using subject-specific h0
    num_subjects = None
    if args.use_subject_h0:
        print("\nDetecting number of subjects...")
        num_subjects = get_num_subjects(args.subj_path)
        print(f"Number of unique subjects: {num_subjects}")
    
    # Model kwargs
    model_kwargs = {
        'hidden_state_dim': args.hidden_state_dim,
        'utility_hidden_dim': args.utility_hidden_dim,
        'rnn_type': args.rnn_type,
        'use_subject_h0': args.use_subject_h0,
        'num_subjects': num_subjects
    }
    
    # Extract Hidden States
    hidden_states_path = os.path.join(output_dir, 'hidden_states.npz')
    
    if args.skip_extraction and os.path.exists(hidden_states_path):
        print("\n Skipping extraction (file already exists)")
        print(f"Using: {hidden_states_path}")
    else:
        print("\nExtracting Hidden States...")
        print("-" * 80)
        
        hidden_data = extract_from_saved_model(
            model_path=args.model_path,
            data_path_obj=args.obj_path,
            data_path_subj=args.subj_path,
            model_kwargs=model_kwargs,
            output_path=hidden_states_path,
            device=device,
            batch_size=args.batch_size
        )
        
        print(f"\nExtracted hidden states:")
        print(f"  Shape: {hidden_data['hidden_states'].shape}")
        print(f"  Saved to: {hidden_states_path}")
    
    # Analyze Hidden States
    if args.skip_analysis:
        print("\nSkipping analysis")
    else:
        print("\nAnalyzing Hidden States...")
        print("-" * 80)
        
        analysis_dir = os.path.join(output_dir, 'analysis')
        
        analysis_results = complete_analysis_pipeline(
            hidden_states_path=hidden_states_path,
            model_path=args.model_path,
            model_kwargs=model_kwargs,
            output_dir=analysis_dir,
            n_clusters=args.n_clusters,
            device_str=args.device
        )
        
        print(f"\nAnalysis results saved to: {analysis_dir}/")
    
    print("\n" + "="*80)
    print("INTERPRETATION COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {hidden_states_path}")
    
    if not args.skip_analysis:
        print(f"  - {analysis_dir}/elbow_method.png")
        print(f"  - {analysis_dir}/pca_by_outcome.png")
        print(f"  - {analysis_dir}/pca_by_trial.png")
        print(f"  - {analysis_dir}/pca_clusters_k*.png")
        print(f"  - {analysis_dir}/utility_by_cluster_k*.png")
        print(f"  - {analysis_dir}/utility_by_cluster_k*_features.png")
        print(f"  - {analysis_dir}/cluster_features_k*.csv")
        
        if analysis_results and 'features' in analysis_results:
            print("\nCluster Summary:")
            print(analysis_results['features'].to_string(index=False))
    
    print("="*80)


if __name__ == "__main__":
    main()