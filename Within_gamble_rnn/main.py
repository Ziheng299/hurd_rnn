import torch
import argparse
import json
import os
from datetime import datetime

from data_preprocessing import preprocess_pipeline
from model import create_model
from train import train_learning_curve
from visualize import plot_learning_curve
import config 


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train Recurrent Neural EU Models')
    
    # Data paths
    parser.add_argument('--obj_path', type=str, default=config.OBJ_PATH,
                       help='Path to c13k_obj_feats_uid.csv')
    parser.add_argument('--subj_path', type=str, default=config.SUBJ_PATH,
                       help='Path to c13k_subject_data_uid.csv')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                       help='Directory to save results')
    
    # Model settings
    parser.add_argument('--model_type', type=str, default='both',
                       choices=['baseline', 'recurrent', 'both'],
                       help='Which model(s) to train')
    parser.add_argument('--hidden_state_dim', type=int, default=config.HIDDEN_STATE_DIM,
                       help='Hidden state dimension for RNN')
    parser.add_argument('--utility_hidden_dim', type=int, default=config.UTILITY_HIDDEN_DIM,
                       help='Hidden layer size in utility network')
    parser.add_argument('--rnn_type', type=str, default=config.RNN_TYPE,
                       choices=['lstm', 'gru'],
                       help='Type of RNN cell')
    parser.add_argument('--use_subject_h0', action='store_true',
                       default=config.USE_SUBJECT_H0,
                       help='Use subject-specific h0')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=config.NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, 
                       default=config.EARLY_STOPPING_PATIENCE,
                       help='Early stopping patience')
    
    # Learning curve settings
    parser.add_argument('--train_percentages', type=float, nargs='+',
                       default=config.TRAIN_PERCENTAGES,
                       help='Percentages of training data to use')
    
    # Other settings
    parser.add_argument('--test_size', type=float, default=config.TEST_SIZE,
                       help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=config.RANDOM_STATE,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--verbose', action='store_true', default=config.VERBOSE,
                       help='Print detailed progress')
    
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


def save_results(results: dict, filepath: str):
    """
    Save results to JSON file.
    """
    serializable_results = {}
    for pct, result in results.items():
        serializable_results[str(pct)] = {
            'percentage': result['percentage'],
            'n_train_samples': result['n_train_samples'],
            'n_test_samples': result['n_test_samples'],
            'final_test_loss': result['history']['final_test_loss'],
            'final_test_accuracy': result['history']['final_test_accuracy'],
            'final_per_trial_accuracy': result['history']['final_per_trial_accuracy']
        }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved results to {filepath}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to {config_path}")
    
    print("\n" + "="*80)
    print("RECURRENT NEURAL EU - TRAINING PIPELINE")
    print("="*80)
    
    data = preprocess_pipeline(
        obj_path=args.obj_path,
        subj_path=args.subj_path,
        test_size=args.test_size,
        random_state=args.random_state,
        batch_size=args.batch_size
    )
    
    train_sequences = data['train_sequences']
    test_sequences = data['test_sequences']
    
    # Get number of subjects
    num_subjects = None
    if args.use_subject_h0:
        all_subject_ids = set()
        for seq in train_sequences:
            all_subject_ids.add(seq['subject_id'])
        num_subjects = len(all_subject_ids)
        print(f"Number of unique subjects: {num_subjects}")
    
    # Training config dict
    train_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'early_stopping_patience': args.early_stopping_patience,
        'random_state': args.random_state,
        'use_regularization': args.use_subject_h0 and config.USE_REGULARIZATION,
        'lambda_l2': config.LAMBDA_L2,
        'verbose': args.verbose
    }
    
    # Train Baseline Model
    baseline_results = None
    if args.model_type in ['baseline', 'both']:
        print("\nTraining Baseline Model...")
        print("-" * 80)
        
        baseline_results = train_learning_curve(
            model_class=lambda **kwargs: create_model(model_type='baseline', **kwargs),
            model_kwargs={
                'utility_hidden_dim': args.utility_hidden_dim,
                'stochastic_spec': config.STOCHASTIC_SPEC
            },
            train_sequences=train_sequences,
            test_sequences=test_sequences,
            percentages=args.train_percentages,
            config=train_config,
            device=device
        )
        
        # Save baseline results
        baseline_path = os.path.join(output_dir, 'baseline_results.json')
        save_results(baseline_results, baseline_path)
        
        # Save best baseline model
        best_pct = 100 if 100 in baseline_results else max(baseline_results.keys())
        best_model = baseline_results[best_pct]['model']
        torch.save(best_model.state_dict(), 
                  os.path.join(output_dir, 'baseline_model.pt'))
        print(f"Saved best baseline model (trained on {best_pct}% data)")
    
    # Train RNN Model
    rnn_results = None
    if args.model_type in ['recurrent', 'both']:
        print("\nTraining Recurrent Model...")
        print("-" * 80)
        
        rnn_results = train_learning_curve(
            model_class=lambda **kwargs: create_model(model_type='recurrent', **kwargs),
            model_kwargs={
                'hidden_state_dim': args.hidden_state_dim,
                'utility_hidden_dim': args.utility_hidden_dim,
                'rnn_type': args.rnn_type,
                'stochastic_spec': config.STOCHASTIC_SPEC,
                'use_subject_h0': args.use_subject_h0,
                'num_subjects': num_subjects
            },
            train_sequences=train_sequences,
            test_sequences=test_sequences,
            percentages=args.train_percentages,
            config=train_config,
            device=device
        )
        
        # Save RNN results
        rnn_path = os.path.join(output_dir, 'rnn_results.json')
        save_results(rnn_results, rnn_path)
        
        # Save best RNN model
        best_pct = 100 if 100 in rnn_results else max(rnn_results.keys())
        best_model = rnn_results[best_pct]['model']
        torch.save(best_model.state_dict(), 
                  os.path.join(output_dir, 'rnn_model.pt'))
        print(f"Saved best RNN model (trained on {best_pct}% data)")
    
    # Visualize Results
    if args.model_type == 'both' and baseline_results and rnn_results:
        print("-" * 80)
        
        # Plot loss comparison
        plot_learning_curve(
            baseline_results=baseline_results,
            rnn_results=rnn_results,
            metric='loss',
            save_path=os.path.join(output_dir, 'learning_curve_loss.png'),
            title='Learning Curve: Test Loss vs Training Data Size'
        )
        
        # Plot accuracy comparison
        plot_learning_curve(
            baseline_results=baseline_results,
            rnn_results=rnn_results,
            metric='accuracy',
            save_path=os.path.join(output_dir, 'learning_curve_accuracy.png'),
            title='Learning Curve: Test Accuracy vs Training Data Size'
        )
        
        print(f"Plots saved to {output_dir}/")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    
    if baseline_results:
        best_baseline_loss = min([r['history']['final_test_loss'] 
                                 for r in baseline_results.values()])
        print(f"Best Baseline Test Loss: {best_baseline_loss:.4f}")
    
    if rnn_results:
        best_rnn_loss = min([r['history']['final_test_loss'] 
                            for r in rnn_results.values()])
        print(f"Best RNN Test Loss: {best_rnn_loss:.4f}")
    
    if baseline_results and rnn_results:
        improvement = (best_baseline_loss - best_rnn_loss) / best_baseline_loss * 100
        print(f"RNN Improvement: {improvement:+.2f}%")
    
    print("="*80)


if __name__ == "__main__":
    main()