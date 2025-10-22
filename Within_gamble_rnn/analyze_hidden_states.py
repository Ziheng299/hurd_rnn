import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from typing import Dict, List
import os
from sklearn.decomposition import PCA

from model import create_model


def load_hidden_states(path: str) -> Dict:
    """Load hidden states from npz file."""
    data = np.load(path)
    result = {key: data[key] for key in data.files}
    print(f"Loaded {result['hidden_states'].shape[0]} sequences")
    return result

def prepare_h_for_analysis(hidden_states: np.ndarray,
                           flatten_trials: bool = True) -> np.ndarray:
    """
    Prepare hidden states for analysis.
    a
    Returns:
        h_matrix: (N*5, H) or (N, H) depending on flatten_trials
    """
    if flatten_trials:
        # Flatten: (N, 5, H) -> (N*5, H)
        N, T, H = hidden_states.shape
        h_matrix = hidden_states.reshape(N * T, H)
        print(f"Flattened h shape: {h_matrix.shape}")
    else:
        # Use only first trial
        h_matrix = hidden_states[:, 0, :]  # (N, H)
        print(f"Using only trial 1, h shape: {h_matrix.shape}")
    
    return h_matrix

def elbow_method(h_matrix: np.ndarray,
                k_range: range = range(2, 21),
                save_path: str = None) -> Dict:
    """
    Perform elbow method to determine optimal number of clusters.
    """
    scaler = StandardScaler()
    h_scaled = scaler.fit_transform(h_matrix)
    
    inertias = []
    silhouette_scores = []
    k_values = list(k_range)
    
    for k in k_values:
        print(f"  Testing k={k}...", end=' ')
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(h_scaled)
        
        inertia = kmeans.inertia_
        sil_score = silhouette_score(h_scaled, labels)
        
        inertias.append(inertia)
        silhouette_scores.append(sil_score)
        
        print(f"Inertia: {inertia:.2f}, Silhouette: {sil_score:.3f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Inertia (elbow)
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette score
    ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score by k', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved elbow plot to {save_path}")
    
    plt.show()
    
    # Find optimal k (max silhouette score)
    optimal_k_idx = np.argmax(silhouette_scores)
    optimal_k = k_values[optimal_k_idx]
    
    print(f"\nRecommended k (max silhouette): {optimal_k}")
    print(f"Silhouette score: {silhouette_scores[optimal_k_idx]:.3f}")
    
    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k
    }


def cluster_and_get_centroids(h_matrix: np.ndarray,
                              n_clusters: int) -> tuple:
    """
    Cluster hidden states and return centroids.
    """
    print(f"\nClustering with k={n_clusters}...")
    
    scaler = StandardScaler()
    h_scaled = scaler.fit_transform(h_matrix)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(h_scaled)
    
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    
    print(f"Cluster sizes: {np.bincount(labels)}")
    
    return labels, centroids_original, scaler


def compute_utility_for_cluster(model,
                                centroid_h: np.ndarray,
                                outcome_range: np.ndarray,
                                device: torch.device) -> np.ndarray:
    """
    Compute utility function for a given cluster centroid.
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        h = torch.tensor(centroid_h, dtype=torch.float32).unsqueeze(0).to(device)  # (1, H)
        
        utilities = []
        for outcome_val in outcome_range:
            outcome_tensor = torch.tensor([[outcome_val]], dtype=torch.float32).to(device)  # (1, 1)
            u = model.utility_net(outcome_tensor, h)
            utilities.append(u.item())
        
        utilities = np.array(utilities)
    
    return utilities


def plot_utility_functions_by_cluster(model,
                                     centroids: np.ndarray,
                                     labels: np.ndarray,
                                     outcome_range: np.ndarray = None,
                                     device: torch.device = None,
                                     save_path: str = None):
    """
    Plot utility functions for each cluster centroid.
    """
    if outcome_range is None:
        outcome_range = np.linspace(-200, 200, 400) 
    
    if device is None:
        device = torch.device('cpu')
    
    n_clusters = centroids.shape[0]
    colors = plt.cm.tab10(np.arange(n_clusters))
    
    plt.figure(figsize=(12, 7))
    
    for cluster_id in range(n_clusters):
        print(f"  Cluster {cluster_id}...", end=' ')
        
        centroid_h = centroids[cluster_id]
        utilities = compute_utility_for_cluster(
            model, centroid_h, outcome_range, device
        )
        
        cluster_size = (labels == cluster_id).sum()
        
        plt.plot(outcome_range, utilities,
                linewidth=2.5, color=colors[cluster_id],
                label=f'Cluster {cluster_id} (n={cluster_size})',
                alpha=0.8)
        
        print(f"Done")
    
    # Add reference lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.xlabel('Objective Outcome Value', fontsize=13)
    plt.ylabel('Utility u(x|h)', fontsize=13)
    plt.title('Utility Functions by Hidden State Cluster', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(-200, 200) 
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved utility plot to {save_path}")
    
    plt.show()


def plot_utility_features_by_cluster(model,
                                    centroids: np.ndarray,
                                    labels: np.ndarray,
                                    device: torch.device = None,
                                    save_path: str = None):
    """
    Analyze and plot utility features (loss aversion, curvature) by cluster.
    """
    if device is None:
        device = torch.device('cpu')
    
    n_clusters = centroids.shape[0]
    outcome_range = np.linspace(-200, 200, 400)
    
    features = []
    
    for cluster_id in range(n_clusters):
        centroid_h = centroids[cluster_id]
        utilities = compute_utility_for_cluster(
            model, centroid_h, outcome_range, device
        )
        
        # 1. Loss aversion: |u(-100)| / |u(+100)|
        u_loss = np.interp(-100, outcome_range, utilities)
        u_gain = np.interp(100, outcome_range, utilities)
        
        if abs(u_gain) > 1e-6:
            loss_aversion = abs(u_loss) / abs(u_gain)
        else:
            loss_aversion = 0
        
        # 2. Curvature at x=50 (second derivative approximation)
        idx_50 = np.argmin(np.abs(outcome_range - 50))
        dx = outcome_range[1] - outcome_range[0]  # spacing
        
        if idx_50 > 0 and idx_50 < len(utilities) - 1:
            # Second derivative ≈ (f(x+dx) - 2f(x) + f(x-dx)) / dx^2
            curvature = (utilities[idx_50+1] - 2*utilities[idx_50] + utilities[idx_50-1]) / (dx**2)
        else:
            curvature = 0
        
        # 3. Reference point (where u(x) crosses 0)
        zero_crossings = np.where(np.diff(np.sign(utilities)))[0]
        if len(zero_crossings) > 0:
            idx = zero_crossings[0]
            x1, x2 = outcome_range[idx], outcome_range[idx+1]
            y1, y2 = utilities[idx], utilities[idx+1]
            ref_point = x1 - y1 * (x2 - x1) / (y2 - y1) 
        else:
            ref_point = 0
        
        cluster_size = (labels == cluster_id).sum()
        
        features.append({
            'cluster': cluster_id,
            'size': cluster_size,
            'loss_aversion': loss_aversion,
            'curvature': curvature,
            'reference_point': ref_point
        })
    
    df = pd.DataFrame(features)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = plt.cm.tab10(np.arange(n_clusters))
    
    # Loss aversion
    axes[0].bar(df['cluster'], df['loss_aversion'], color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Cluster', fontsize=11)
    axes[0].set_ylabel('Loss Aversion Ratio', fontsize=11)
    axes[0].set_title('Loss Aversion by Cluster\n|u(-100)| / |u(+100)|', fontsize=12, fontweight='bold')
    axes[0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Symmetric (LA=1)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticks(df['cluster'])
    
    # Curvature (negative = risk averse, positive = risk seeking)
    axes[1].bar(df['cluster'], df['curvature'], color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Cluster', fontsize=11)
    axes[1].set_ylabel('Curvature at x=50', fontsize=11)
    axes[1].set_title('Risk Attitude by Cluster\n(- = risk averse, + = risk seeking)', 
                     fontsize=12, fontweight='bold')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Risk Neutral')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(df['cluster'])
    
    # Reference point
    axes[2].bar(df['cluster'], df['reference_point'], color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Cluster', fontsize=11)
    axes[2].set_ylabel('Reference Point (x where u=0)', fontsize=11)
    axes[2].set_title('Reference Point by Cluster', fontsize=12, fontweight='bold')
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Reference')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_xticks(df['cluster'])
    
    plt.tight_layout()
    
    if save_path:
        features_path = save_path.replace('.png', '_features.png')
        os.makedirs(os.path.dirname(features_path) or '.', exist_ok=True)
        plt.savefig(features_path, dpi=300, bbox_inches='tight')
        print(f"Saved features plot to {features_path}")
    
    plt.show()
    
    print("\nUtility Features by Cluster:")
    print(df.to_string(index=False))
    
    return df

def pca_analysis(h_matrix: np.ndarray,
                n_components: int = 2,
                standardize: bool = True) -> tuple:
    """
    Perform PCA on hidden states.
    """
    if standardize:
        scaler = StandardScaler()
        h_scaled = scaler.fit_transform(h_matrix)
    else:
        h_scaled = h_matrix
    
    pca = PCA(n_components=n_components)
    h_reduced = pca.fit_transform(h_scaled)
    
    explained_var = pca.explained_variance_ratio_
    print(f"Explained variance: {explained_var}")
    print(f"Total explained: {explained_var.sum():.2%}")
    
    return h_reduced, pca, explained_var


def plot_pca_by_outcome(h_reduced: np.ndarray,
                       contexts: np.ndarray,
                       save_path: str = None):
    """
    Plot PCA colored by outcome (win/loss).
    """
    N, T, _ = contexts.shape
    contexts_flat = contexts.reshape(N * T, -1)
    
    rewards = contexts_flat[:, 1]
    outcome_labels = np.where(rewards > 0, 'Win', np.where(rewards < 0, 'Loss', 'Neutral'))
    
    plt.figure(figsize=(10, 8))
    
    for outcome, color in [('Win', 'green'), ('Loss', 'red'), ('Neutral', 'gray')]:
        mask = outcome_labels == outcome
        plt.scatter(h_reduced[mask, 0], h_reduced[mask, 1],
                   alpha=0.3, s=10, c=color, label=outcome)
    
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title('Hidden States (PCA) Colored by Outcome', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_pca_by_trial(h_reduced: np.ndarray,
                     N_sequences: int,
                     save_path: str = None):
    """
    Plot PCA colored by trial number.
    """
    trial_labels = np.repeat(np.arange(1, 6), N_sequences)
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for trial in range(1, 6):
        mask = trial_labels == trial
        plt.scatter(h_reduced[mask, 0], h_reduced[mask, 1],
                   alpha=0.4, s=10, c=[colors[trial-1]], label=f'Trial {trial}')
    
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title('Hidden States (PCA) Colored by Trial Number', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_clusters(h_reduced: np.ndarray,
                 labels: np.ndarray,
                 save_path: str = None):
    """
    Plot PCA colored by cluster assignment.
    """
    plt.figure(figsize=(10, 8))
    
    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab10(np.arange(n_clusters))
    
    for cluster in range(n_clusters):
        mask = labels == cluster
        plt.scatter(h_reduced[mask, 0], h_reduced[mask, 1],
                   alpha=0.4, s=10, c=[colors[cluster]], 
                   label=f'Cluster {cluster} (n={mask.sum()})')
    
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title('Hidden States Clustered (K-means)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def complete_analysis_pipeline(hidden_states_path: str,
                              model_path: str,
                              model_kwargs: Dict,
                              output_dir: str = "./analysis",
                              n_clusters: int = None,
                              device_str: str = "cpu"):
    """
    Complete analysis: PCA + Elbow + Utility visualization.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("COMPLETE HIDDEN STATE ANALYSIS")
    print("="*80)
    
    data = load_hidden_states(hidden_states_path)
    h = data['hidden_states']  # (N, 5, H)
    contexts = data['contexts']
    N_sequences = h.shape[0]
    h_matrix = prepare_h_for_analysis(h, flatten_trials=True)
    
    # PCA Analysis
    print("\n" + "="*80)
    print("PCA ANALYSIS")
    print("="*80)
    
    h_pca, pca_obj, explained_var = pca_analysis(h_matrix, n_components=2)
    
    plot_pca_by_outcome(
        h_pca, contexts,
        save_path=os.path.join(output_dir, 'pca_by_outcome.png')
    )
    
    plot_pca_by_trial(
        h_pca, N_sequences,
        save_path=os.path.join(output_dir, 'pca_by_trial.png')
    )
    
    # Elbow Method
    print("\n" + "="*80)
    print("ELBOW METHOD FOR CLUSTERING")
    print("="*80)
    
    elbow_results = elbow_method(
        h_matrix,
        k_range=range(2, 21),
        save_path=os.path.join(output_dir, 'elbow_method.png')
    )
    
    # Determine k
    if n_clusters is None:
        n_clusters = elbow_results['optimal_k']
        print(f"\nUsing optimal k={n_clusters} from elbow method")
    else:
        print(f"\nUsing specified k={n_clusters}")
    
    # Clustering
    print("\n" + "="*80)
    print("CLUSTERING")
    print("="*80)
    
    labels, centroids, scaler = cluster_and_get_centroids(h_matrix, n_clusters)
    
    plot_clusters(
        h_pca, labels,
        save_path=os.path.join(output_dir, f'pca_clusters_k{n_clusters}.png')
    )
    
    # Utility Analysis
    print("\n" + "="*80)
    print("UTILITY FUNCTION ANALYSIS")
    print("="*80)
    
    model = create_model(model_type='recurrent', **model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Plot utility functions
    plot_utility_functions_by_cluster(
        model=model,
        centroids=centroids,
        labels=labels,
        device=device,
        save_path=os.path.join(output_dir, f'utility_by_cluster_k{n_clusters}.png')
    )
    
    # Analyze features
    features_df = plot_utility_features_by_cluster(
        model=model,
        centroids=centroids,
        labels=labels,
        device=device,
        save_path=os.path.join(output_dir, f'utility_by_cluster_k{n_clusters}.png')
    )
    
    # Save results
    np.savez(os.path.join(output_dir, f'analysis_results_k{n_clusters}.npz'),
             h_pca=h_pca,
             explained_variance=explained_var,
             centroids=centroids,
             labels=labels,
             cluster_sizes=np.bincount(labels))
    
    features_df.to_csv(os.path.join(output_dir, f'cluster_features_k{n_clusters}.csv'), index=False)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All results saved to {output_dir}/")
    
    return {
        'h_pca': h_pca,
        'pca_object': pca_obj,
        'n_clusters': n_clusters,
        'centroids': centroids,
        'labels': labels,
        'features': features_df
    }