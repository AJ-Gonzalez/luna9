"""
Visualization tools for geometric clustering and relationship inference.

Creates plots and dashboards to understand clustering results.
"""

import numpy as np
import polars as pl
from typing import Dict, Optional, List
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from .clustering import SemanticRelationship

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def plot_clusters_2d(
    features_2d: np.ndarray,
    labels: np.ndarray,
    validation: Dict,
    output_path: Optional[Path] = None,
    title: str = "Message Pairs Clustered by Geometric Properties"
) -> None:
    """
    Create 2D scatter plot of geometric clusters with relationship labels.

    Args:
        features_2d: Array of shape (n_samples, 2) with reduced features
        labels: Cluster labels for each sample
        validation: Validation dict from validate_clusters()
        output_path: Where to save the plot (None = display only)
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Get relationship signatures
    signatures = validation.get('relationship_signatures', {})

    # Left plot: Scatter colored by cluster
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        if label == -1:  # Noise cluster
            color = 'gray'
            marker = 'x'
            label_text = 'Noise'
        else:
            color = colors[idx]
            marker = 'o'
            rel_type = signatures.get(label, SemanticRelationship.UNKNOWN)
            label_text = f"C{label}: {rel_type.value}"

        mask = labels == label
        axes[0].scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            marker=marker,
            label=label_text,
            alpha=0.6,
            s=40,
            edgecolors='black',
            linewidths=0.5
        )

    axes[0].set_title(title)
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right plot: Cluster profiles table
    axes[1].axis('off')

    profile_text = "Cluster Profiles (Geometric Signatures)\n"
    profile_text += "=" * 60 + "\n\n"

    for cluster_id, profile in validation.get('cluster_profiles', {}).items():
        rel_type = signatures.get(cluster_id, SemanticRelationship.UNKNOWN)

        profile_text += f"Cluster {cluster_id}: {rel_type.value.upper()}\n"
        profile_text += f"  Count: {profile.get('count', 0)} pairs\n"

        # Key metrics
        if 'median_normalized_curvature' in profile:
            profile_text += f"  Curvature: {profile['median_normalized_curvature']:.4f}\n"
        if 'median_cosine_similarity' in profile:
            profile_text += f"  Similarity: {profile['median_cosine_similarity']:.3f}\n"
        if 'median_influence_overlap' in profile:
            profile_text += f"  Overlap: {profile['median_influence_overlap']:.3f}\n"
        if 'median_geodesic_distance' in profile:
            profile_text += f"  Distance: {profile['median_geodesic_distance']:.3f}\n"

        profile_text += "\n"

    axes[1].text(
        0.05, 0.95,
        profile_text,
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment='top',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_distributions(
    features_df: pl.DataFrame,
    features_to_plot: Optional[List[str]] = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot distributions of key geometric features.

    Args:
        features_df: DataFrame with geometric features
        features_to_plot: List of feature names to plot (None = all numeric)
        output_path: Where to save the plot
    """
    if features_to_plot is None:
        # Default key features
        features_to_plot = [
            'geodesic_distance',
            'total_path_curvature',
            'cosine_similarity',
            'influence_overlap',
        ]

    # Filter to available features
    features_to_plot = [f for f in features_to_plot if f in features_df.columns]

    if not features_to_plot:
        logger.warning("No features available to plot")
        return

    n_features = len(features_to_plot)
    n_cols = 2
    n_rows = (n_features + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features_to_plot):
        data = features_df[feature].drop_nulls().to_numpy()

        if len(data) == 0:
            continue

        axes[idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{feature} Distribution')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Count')
        axes[idx].grid(True, alpha=0.3)

        # Add mean/median lines
        mean_val = np.mean(data)
        median_val = np.median(data)
        axes[idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        axes[idx].axvline(median_val, color='blue', linestyle='--', label=f'Median: {median_val:.3f}')
        axes[idx].legend()

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature distributions to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_cluster_comparison(
    features_df: pl.DataFrame,
    cluster_labels: np.ndarray,
    features_to_compare: Optional[List[str]] = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Create box plots comparing feature distributions across clusters.

    Args:
        features_df: DataFrame with geometric features
        cluster_labels: Cluster assignment for each sample
        features_to_compare: Features to compare across clusters
        output_path: Where to save the plot
    """
    if features_to_compare is None:
        features_to_compare = [
            'normalized_curvature',
            'cosine_similarity',
            'influence_overlap',
            'geodesic_distance',
        ]

    # Filter to available features
    features_to_compare = [f for f in features_to_compare if f in features_df.columns]

    if not features_to_compare:
        logger.warning("No features available for comparison")
        return

    # Add cluster labels to dataframe
    features_df = features_df.with_columns(
        pl.Series('cluster', cluster_labels)
    )

    n_features = len(features_to_compare)
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features_to_compare):
        # Convert to pandas for seaborn
        plot_data = features_df.select(['cluster', feature]).to_pandas()
        plot_data = plot_data.dropna()

        sns.boxplot(
            data=plot_data,
            x='cluster',
            y=feature,
            ax=axes[idx]
        )

        axes[idx].set_title(f'{feature} by Cluster')
        axes[idx].set_xlabel('Cluster')
        axes[idx].set_ylabel(feature)
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster comparison to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_3d_clusters(
    features_3d: np.ndarray,
    labels: np.ndarray,
    validation: Dict,
    output_path: Optional[Path] = None,
    title: str = "3D Geometric Clusters"
) -> None:
    """
    Create 3D scatter plot of clusters (requires 3D reduction).

    Args:
        features_3d: Array of shape (n_samples, 3)
        labels: Cluster labels
        validation: Validation dict
        output_path: Where to save
        title: Plot title
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    signatures = validation.get('relationship_signatures', {})
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        if label == -1:
            color = 'gray'
            marker = 'x'
            label_text = 'Noise'
        else:
            color = colors[idx]
            marker = 'o'
            rel_type = signatures.get(label, SemanticRelationship.UNKNOWN)
            label_text = f"C{label}: {rel_type.value}"

        mask = labels == label
        ax.scatter(
            features_3d[mask, 0],
            features_3d[mask, 1],
            features_3d[mask, 2],
            c=[color],
            marker=marker,
            label=label_text,
            alpha=0.6,
            s=40
        )

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved 3D cluster plot to {output_path}")
    else:
        plt.show()

    plt.close()


def create_dashboard(
    features_df: pl.DataFrame,
    features_2d: np.ndarray,
    cluster_labels: np.ndarray,
    validation: Dict,
    output_dir: Path
) -> None:
    """
    Create complete visualization dashboard with all plots.

    Args:
        features_df: DataFrame with geometric features
        features_2d: 2D reduced features
        cluster_labels: Cluster assignments
        validation: Validation results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating visualization dashboard...")

    # 1. Main cluster plot
    plot_clusters_2d(
        features_2d,
        cluster_labels,
        validation,
        output_path=output_dir / 'clusters_2d.png'
    )

    # 2. Feature distributions
    plot_feature_distributions(
        features_df,
        output_path=output_dir / 'feature_distributions.png'
    )

    # 3. Cluster comparison
    plot_cluster_comparison(
        features_df,
        cluster_labels,
        output_path=output_dir / 'cluster_comparison.png'
    )

    logger.info(f"Dashboard saved to {output_dir}")
