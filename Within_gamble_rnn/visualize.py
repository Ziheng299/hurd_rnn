import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os

def plot_learning_curve(baseline_results: Dict,
                        rnn_results: Dict,
                        metric: str = "loss",
                        save_path: Optional[str] = None,
                        title: Optional[str] = None):
    """
    Plot learning curve comparing baseline vs RNN models.
    """
    if metric == "loss":
        key = "final_test_loss"
        ylabel = "Test BCE Loss"
        title = title or "Learning Curve: Test Loss vs Training Data Size"
    elif metric == "accuracy":
        key = "final_test_accuracy"
        ylabel = "Test Accuracy"
        title = title or "Learning Curve: Test Accuracy vs Training Data Size"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    pcts = sorted(baseline_results.keys())

    baseline_values = [baseline_results[p]['history'][key] for p in pcts]
    rnn_values      = [rnn_results[p]['history'][key]      for p in pcts]

    plt.figure(figsize=(10, 6))
    plt.plot(pcts, baseline_values, marker='o', linewidth=2, markersize=8,
             label='Baseline (Neural EU)', color='#1f77b4')
    plt.plot(pcts, rnn_values, marker='s', linewidth=2, markersize=8,
             label='RNN (Recurrent Neural EU)', color='#ff7f0e')

    plt.xlabel('Percent of Training Data Used (%)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(pcts)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath: 
            os.makedirs(dirpath, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()
