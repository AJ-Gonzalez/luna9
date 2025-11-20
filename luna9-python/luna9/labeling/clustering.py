"""
Clustering and validation for geometric semantic inference.

Implements dimensionality reduction, clustering, and validation
of geometric signatures against semantic relationship types.
"""

import numpy as np
import polars as pl
from typing import Dict, Tuple, Optional, List
from enum import Enum
import logging

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")

logger = logging.getLogger(__name__)


class SemanticRelationship(Enum):
    """
    Types of semantic relationships between message pairs.

    These are the target labels we're trying to infer from geometric properties.
    """
    NECESSARY = "necessary"          # Direct Q&A, prerequisite knowledge
    OPPOSED = "opposed"              # Contradictory, conflicting views
    ANCILLARY = "ancillary"          # Supporting, optional context
    HIERARCHICAL = "hierarchical"    # Parent-child, concept levels
    IRRELEVANT = "irrelevant"        # Unrelated topics
    UNKNOWN = "unknown"              # Cannot determine from geometry


def normalize_features(
    features_df: pl.DataFrame,
    method: str = 'robust',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pl.DataFrame, object]:
    """
    Normalize geometric features for clustering.

    Args:
        features_df: DataFrame with geometric features
        method: 'robust' (handles outliers) or 'standard' (assumes normal distribution)
        exclude_cols: Columns to exclude from normalization (e.g., metadata)

    Returns:
        Tuple of (normalized DataFrame, scaler object)
    """
    if exclude_cols is None:
        exclude_cols = ['msg1', 'msg2', 'conversation_id', 'distance',
                       'pair_type', 'cluster', 'error']

    # Identify feature columns
    feature_cols = [col for col in features_df.columns
                   if col not in exclude_cols and features_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    if not feature_cols:
        logger.warning("No numeric feature columns found for normalization")
        return features_df, None

    logger.info(f"Normalizing {len(feature_cols)} features using {method} scaling")

    # Convert to pandas for sklearn
    df_pandas = features_df.to_pandas()

    # Select scaler
    if method == 'robust':
        scaler = RobustScaler()  # Better with outliers
    else:
        scaler = StandardScaler()

    # Fit and transform
    df_pandas[feature_cols] = scaler.fit_transform(df_pandas[feature_cols])

    # Convert back to polars
    return pl.from_pandas(df_pandas), scaler


def reduce_dimensions(
    features_array: np.ndarray,
    n_components: int = 2,
    method: str = 'umap',
    pca_preprocess: bool = True,
    pca_components: int = 10
) -> Tuple[np.ndarray, Dict]:
    """
    Reduce high-dimensional geometric features to 2D/3D for visualization and clustering.

    Uses two-stage approach:
    1. PCA to reduce noise and dimensionality
    2. UMAP to preserve local structure (or PCA if UMAP unavailable)

    Args:
        features_array: Array of shape (n_samples, n_features)
        n_components: Target dimensionality (typically 2 or 3)
        method: 'umap' or 'pca'
        pca_preprocess: Whether to apply PCA before UMAP
        pca_components: Number of PCA components if preprocessing

    Returns:
        Tuple of (reduced features, reducer dict with 'pca' and 'final' keys)
    """
    reducers = {}

    # Stage 1: PCA preprocessing (optional but recommended)
    if pca_preprocess and features_array.shape[1] > pca_components:
        logger.info(f"Stage 1: PCA to {pca_components} components")
        pca = PCA(n_components=pca_components)
        features_pca = pca.fit_transform(features_array)
        reducers['pca'] = pca

        explained_var = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA explained variance: {explained_var:.2%}")
    else:
        features_pca = features_array
        reducers['pca'] = None

    # Stage 2: Final reduction
    if method == 'umap' and HAS_UMAP:
        logger.info(f"Stage 2: UMAP to {n_components} components")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,  # Preserve local structure
            min_dist=0.1,    # Minimum distance between points
            metric='euclidean',
            random_state=42
        )
        features_reduced = reducer.fit_transform(features_pca)
        reducers['final'] = reducer

    else:
        if method == 'umap' and not HAS_UMAP:
            logger.warning("UMAP not available, falling back to PCA")

        logger.info(f"Stage 2: PCA to {n_components} components")
        pca_final = PCA(n_components=n_components)
        features_reduced = pca_final.fit_transform(features_pca)
        reducers['final'] = pca_final

        explained_var = pca_final.explained_variance_ratio_.sum()
        logger.info(f"Final PCA explained variance: {explained_var:.2%}")

    return features_reduced, reducers


def cluster_by_geometry(
    features_array: np.ndarray,
    n_clusters: int = 5,
    method: str = 'kmeans',
    **kwargs
) -> Dict:
    """
    Cluster message pairs by geometric similarity.

    Args:
        features_array: Normalized feature array of shape (n_samples, n_features)
        n_clusters: Number of clusters (for k-means)
        method: 'kmeans' or 'dbscan'
        **kwargs: Additional arguments for clustering algorithm

    Returns:
        Dict with:
            - labels: Cluster labels for each sample
            - clusterer: Fitted clustering object
            - silhouette_score: Cluster quality metric (higher = better)
            - davies_bouldin_score: Cluster quality metric (lower = better)
            - n_clusters: Actual number of clusters found
    """
    logger.info(f"Clustering {features_array.shape[0]} samples using {method}")

    if method == 'kmeans':
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            **kwargs
        )
        labels = clusterer.fit_predict(features_array)

    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(features_array)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Compute cluster quality metrics
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters_found > 1:
        try:
            sil_score = silhouette_score(features_array, labels)
            db_score = davies_bouldin_score(features_array, labels)
        except Exception as e:
            logger.warning(f"Could not compute quality metrics: {e}")
            sil_score = np.nan
            db_score = np.nan
    else:
        logger.warning("Only one cluster found, cannot compute quality metrics")
        sil_score = np.nan
        db_score = np.nan

    logger.info(f"Found {n_clusters_found} clusters")
    logger.info(f"Silhouette score: {sil_score:.3f} (higher = better)")
    logger.info(f"Davies-Bouldin score: {db_score:.3f} (lower = better)")

    return {
        'labels': labels,
        'clusterer': clusterer,
        'silhouette_score': sil_score,
        'davies_bouldin_score': db_score,
        'n_clusters': n_clusters_found
    }


def infer_relationship_type(signature: Dict) -> SemanticRelationship:
    """
    Infer semantic relationship type from geometric signature.

    Uses heuristic decision tree based on key geometric properties:
    - path_curvature: Complexity of semantic connection
    - cosine_similarity: Proximity in embedding space
    - influence_overlap: Shared control points
    - geodesic_distance: Actual distance on surface

    Decision boundaries based on empirical observations:
    - NECESSARY: Low curvature (<0.015), high similarity (>0.7)
    - OPPOSED: High curvature (>0.03), low similarity (<0.3), low overlap (<0.3)
    - HIERARCHICAL: Medium curvature (0.015-0.025), high similarity (>0.4)
    - ANCILLARY: Medium similarity (0.4-0.7), medium curvature
    - IRRELEVANT: Low similarity (<0.3), high distance

    Args:
        signature: Dict with median geometric features for a cluster

    Returns:
        SemanticRelationship enum value
    """
    # Extract key metrics (use .get() for safety)
    curv = signature.get('median_total_path_curvature', signature.get('median_normalized_curvature', 0))
    cos_sim = signature.get('median_cosine_similarity', 0)
    influence = signature.get('median_influence_overlap', 0.5)
    geo_dist = signature.get('median_geodesic_distance', 0)

    # Decision tree
    if cos_sim > 0.7 and curv < 0.015:
        # High similarity, low curvature = direct connection
        return SemanticRelationship.NECESSARY

    elif cos_sim < 0.3 and curv > 0.03:
        # Low similarity, high curvature
        if influence < 0.3:
            # Low overlap = opposed views (same topic, different stance)
            return SemanticRelationship.OPPOSED
        else:
            # Not opposed, just distant topics
            return SemanticRelationship.IRRELEVANT

    elif 0.4 < cos_sim < 0.7 and 0.015 < curv < 0.025:
        # Medium similarity and curvature = hierarchical
        return SemanticRelationship.HIERARCHICAL

    elif cos_sim > 0.4:
        # Moderate similarity = ancillary/supporting
        return SemanticRelationship.ANCILLARY

    else:
        # Can't determine confidently
        return SemanticRelationship.UNKNOWN


def validate_clusters(
    features_df: pl.DataFrame,
    cluster_labels: np.ndarray,
    ground_truth_labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Validate clustering results and infer semantic relationships.

    Args:
        features_df: DataFrame with geometric features
        cluster_labels: Cluster assignment for each sample
        ground_truth_labels: Optional ground truth for validation

    Returns:
        Dict with:
            - cluster_profiles: Median feature values for each cluster
            - relationship_signatures: Inferred relationship type for each cluster
            - validation_metrics: Optional metrics if ground truth provided
    """
    validation = {
        'cluster_profiles': {},
        'relationship_signatures': {},
    }

    # Add cluster labels to dataframe
    features_df = features_df.with_columns(
        pl.Series('cluster', cluster_labels)
    )

    # Profile each cluster
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:  # Skip noise cluster from DBSCAN
            continue

        cluster_data = features_df.filter(pl.col('cluster') == cluster_id)

        # Compute median values for signature
        signature = {
            'cluster_id': cluster_id,
            'count': cluster_data.height,
        }

        # Add median of key geometric features
        key_features = [
            'total_path_curvature', 'normalized_curvature',
            'cosine_similarity', 'geodesic_distance',
            'influence_overlap', 'arc_length',
            'avg_gaussian_curvature', 'avg_mean_curvature'
        ]

        for feature in key_features:
            if feature in cluster_data.columns:
                try:
                    median_val = cluster_data[feature].median()
                    signature[f'median_{feature}'] = median_val
                except Exception:
                    pass

        validation['cluster_profiles'][cluster_id] = signature

        # Infer relationship type
        inferred_type = infer_relationship_type(signature)
        validation['relationship_signatures'][cluster_id] = inferred_type

        logger.info(f"Cluster {cluster_id}: {inferred_type.value} ({signature['count']} pairs)")

    # If ground truth provided, compute validation metrics
    if ground_truth_labels is not None:
        validation['validation_metrics'] = compute_validation_metrics(
            cluster_labels,
            ground_truth_labels
        )

    return validation


def compute_validation_metrics(
    predicted_labels: np.ndarray,
    ground_truth_labels: np.ndarray
) -> Dict:
    """
    Compute validation metrics if ground truth labels are available.

    Args:
        predicted_labels: Cluster assignments
        ground_truth_labels: True relationship labels

    Returns:
        Dict with accuracy, precision, recall, etc.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    metrics = {
        'adjusted_rand_index': adjusted_rand_score(ground_truth_labels, predicted_labels),
        'normalized_mutual_info': normalized_mutual_info_score(ground_truth_labels, predicted_labels),
    }

    logger.info(f"Validation metrics: ARI={metrics['adjusted_rand_index']:.3f}, "
               f"NMI={metrics['normalized_mutual_info']:.3f}")

    return metrics
