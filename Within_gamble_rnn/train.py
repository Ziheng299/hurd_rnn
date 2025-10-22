import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from data_preprocessing import DecisionMakingDataset, sample_train_data, get_dataloaders


def compute_loss(predictions: torch.Tensor, 
                 targets: torch.Tensor, 
                 loss_type: str = "bce") -> torch.Tensor:
    """
    Compute loss for choice predictions.
    predictions: P(A) in [0,1], shape (B, T)
    targets:     0=A, 1=B,      shape (B, T)
    """
    if loss_type == "bce":
        targets_for_bce = 1.0 - targets.float()  # 1 if A, 0 if B
        preds = predictions.clamp(min=1e-6, max=1.0 - 1e-6)

        if preds.shape != targets_for_bce.shape:
            raise ValueError(f"Shape mismatch: preds {preds.shape} vs targets {targets_for_bce.shape}")

        return nn.functional.binary_cross_entropy(preds, targets_for_bce, reduction="mean")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute choice accuracy.
    """
    predicted_choices = (predictions < 0.5).long()
    correct = (predicted_choices == targets).float()
    return correct.mean().item()


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                use_regularization: bool = False,
                lambda_l2: float = 1e-4) -> Dict:
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        output = model(batch)
        predictions = output['predictions']       
        targets = batch['choices']               

        loss = compute_loss(predictions, targets)

        # Add regularization if available
        if use_regularization and hasattr(model, "subject_h0") and hasattr(model.subject_h0, "regularization_loss"):
            loss = loss + model.subject_h0.regularization_loss(lambda_l2=lambda_l2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += compute_accuracy(predictions.detach(), targets)
        num_batches += 1

    return {'loss': total_loss / num_batches, 'accuracy': total_accuracy / num_batches}



def evaluate(model: nn.Module,
            dataloader: DataLoader,
            device: torch.device) -> Dict:
    """
    Evaluate model on a dataset.
    """
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    # Per-trial accuracy
    per_trial_correct = torch.zeros(5)
    per_trial_total = torch.zeros(5)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            output = model(batch)
            predictions = output['predictions']
            targets = batch['choices']
            
            loss = compute_loss(predictions, targets)
            total_loss += loss.item()
            total_accuracy += compute_accuracy(predictions, targets)
            
            # Per-trial accuracy
            predicted_choices = (predictions < 0.5).long()
            for t in range(5):
                correct = (predicted_choices[:, t] == targets[:, t]).float()
                per_trial_correct[t] += correct.sum().item()
                per_trial_total[t] += targets.shape[0]
            
            num_batches += 1
    
    per_trial_accuracy = (per_trial_correct / per_trial_total).tolist()
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'per_trial_accuracy': per_trial_accuracy
    }


def train_model(model: nn.Module,
               train_loader: DataLoader,
               test_loader: DataLoader,
               num_epochs: int,
               learning_rate: float,
               device: torch.device,
               use_regularization: bool = False,
               lambda_l2: float = 1e-4,
               early_stopping_patience: int = 10,
               verbose: bool = True) -> Dict:
    """
    Complete training loop.
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_per_trial_accuracy': []
    }
    
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            use_regularization=use_regularization,
            lambda_l2=lambda_l2
        )
        test_metrics = evaluate(model, test_loader, device)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        history['test_per_trial_accuracy'].append(test_metrics['per_trial_accuracy'])
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Test Loss: {test_metrics['loss']:.4f}, "
                  f"Test Acc: {test_metrics['accuracy']:.4f}")
        
        # Early stopping
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
    # Final evaluation
    final_test_metrics = evaluate(model, test_loader, device)
    history['final_test_loss'] = final_test_metrics['loss']
    history['final_test_accuracy'] = final_test_metrics['accuracy']
    history['final_per_trial_accuracy'] = final_test_metrics['per_trial_accuracy']
    
    return history


def train_with_percentage(model_class,
                         model_kwargs: Dict,
                         train_sequences: List[Dict],
                         test_sequences: List[Dict],
                         percentage: float,
                         config: Dict,
                         device: torch.device) -> Dict:
    """
    Train model with a percentage of training data.
    Used for learning curve experiments.
    """
    sampled_train = sample_train_data(
        train_sequences, 
        percentage=percentage,
        random_state=config.get('random_state', 42)
    )
    
    train_loader, test_loader = get_dataloaders(
        sampled_train,
        test_sequences,
        batch_size=config.get('batch_size', 64)
    )
    
    model = model_class(**model_kwargs)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=config.get('num_epochs', 100),
        learning_rate=config.get('learning_rate', 0.001),
        device=device,
        use_regularization=config.get('use_regularization', False),
        lambda_l2=config.get('lambda_l2', 1e-4),
        early_stopping_patience=config.get('early_stopping_patience', 10),
        verbose=config.get('verbose', True)
    )
    
    return {
        'percentage': percentage,
        'history': history,
        'model': model,
        'n_train_samples': len(sampled_train),
        'n_test_samples': len(test_sequences)
    }


def train_learning_curve(model_class,
                        model_kwargs: Dict,
                        train_sequences: List[Dict],
                        test_sequences: List[Dict],
                        percentages: List[float],
                        config: Dict,
                        device: torch.device) -> Dict:
    """
    Train models with different percentages of training data.
    Generates learning curve.
    """
    results = {}
    
    for pct in percentages:
        print(f"\n{'='*60}")
        print(f"Training with {pct}% of training data")
        print(f"{'='*60}")
        
        result = train_with_percentage(
            model_class=model_class,
            model_kwargs=model_kwargs,
            train_sequences=train_sequences,
            test_sequences=test_sequences,
            percentage=pct,
            config=config,
            device=device
        )
        
        results[pct] = result
        
        print(f"\nResults for {pct}%:")
        print(f"  Final Test Loss: {result['history']['final_test_loss']:.4f}")
        print(f"  Final Test Accuracy: {result['history']['final_test_accuracy']:.4f}")
    
    return results


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int,
                   history: Dict,
                   filepath: str):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model: nn.Module,
                   optimizer: Optional[optim.Optimizer],
                   filepath: str,
                   device: torch.device) -> Tuple[nn.Module, Optional[optim.Optimizer], int, Dict]:
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    history = checkpoint.get('history', {})
    
    return model, optimizer, epoch, history