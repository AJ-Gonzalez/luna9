"""
Main orchestration pipeline for geometric relationship inference.

Coordinates the full workflow: Load → Sample → Embed → Extract → Cluster → Validate → Visualize
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json
import logging
import time

# Luna Nine imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "luna9-python"))

from luna9.semantic_surface import SemanticSurface
from sentence_transformers import SentenceTransformer

# Local imports
from .parquet_utils import (
    load_conversations,
    sample_pairs,
    get_parquet_stats,
    validate_data_quality
)
from .geometry_extractor import GeometricFeatureExtractor, get_feature_names
from .clustering import (
    normalize_features,
    reduce_dimensions,
    cluster_by_geometry,
    validate_clusters
)
from .visualization import create_dashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeometricLabelingPipeline:
    """
    End-to-end pipeline for geometric relationship inference.

    Workflow:
    1. Load conversations from Parquet files
    2. Sample message pairs (sequential, distant, random)
    3. Compute embeddings using sentence-transformers
    4. Build semantic surface from embeddings
    5. Extract geometric features for each pair
    6. Normalize and reduce dimensionality
    7. Cluster by geometric similarity
    8. Validate and infer relationship types
    9. Create visualizations
    10. Save labeled dataset
    """

    def __init__(
        self,
        parquet_dir: str,
        output_dir: str = './geometric_labels',
        encoder_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize pipeline.

        Args:
            parquet_dir: Directory containing parquet files
            output_dir: Where to save results
            encoder_model: Sentence-transformers model name
        """
        self.parquet_dir = Path(parquet_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.encoder_model_name = encoder_model
        self.encoder = None
        self.surface = None

        logger.info(f"Initialized pipeline with parquet_dir={parquet_dir}, output_dir={output_dir}")

    def run_full_pipeline(
        self,
        sample_size: int = 5000,
        pair_strategy: str = 'mixed',
        pairs_per_conversation: int = 5,
        max_pairs: int = 5000,
        n_clusters: int = 5,
        clustering_method: str = 'kmeans',
        dimensionality_method: str = 'umap'
    ) -> Dict:
        """
        Execute complete labeling pipeline.

        Args:
            sample_size: Number of conversations to sample
            pair_strategy: How to select pairs ('sequential', 'distant', 'random', 'mixed')
            pairs_per_conversation: Pairs to extract per conversation
            max_pairs: Maximum total pairs
            n_clusters: Number of clusters for k-means
            clustering_method: 'kmeans' or 'dbscan'
            dimensionality_method: 'umap' or 'pca'

        Returns:
            Dict with pipeline results and metrics
        """
        results = {
            'timestamp': time.time(),
            'config': {
                'sample_size': sample_size,
                'pair_strategy': pair_strategy,
                'max_pairs': max_pairs,
                'n_clusters': n_clusters,
                'clustering_method': clustering_method,
                'encoder_model': self.encoder_model_name,
            },
            'steps': {}
        }

        # Step 1: Load and validate data
        logger.info("=" * 60)
        logger.info("STEP 1: Loading and sampling conversations")
        logger.info("=" * 60)

        conversations_df = self._load_and_sample(sample_size)
        results['steps']['sampling'] = {
            'total_messages': conversations_df.height,
            'unique_conversations': conversations_df.select('conversation_id').n_unique(),
        }

        # Step 2: Extract message pairs
        logger.info("=" * 60)
        logger.info("STEP 2: Extracting message pairs")
        logger.info("=" * 60)

        pairs = sample_pairs(
            conversations_df,
            pair_strategy=pair_strategy,
            pairs_per_conversation=pairs_per_conversation,
            max_pairs=max_pairs
        )
        results['steps']['pair_extraction'] = {
            'total_pairs': len(pairs),
            'pair_strategy': pair_strategy,
        }

        # Step 3: Build surface and extract features
        logger.info("=" * 60)
        logger.info("STEP 3: Computing embeddings and building semantic surface")
        logger.info("=" * 60)

        features_df = self._extract_features(pairs)
        results['steps']['feature_extraction'] = {
            'n_pairs': features_df.height,
            'n_features': len(get_feature_names()),
        }

        # Step 4: Normalize features
        logger.info("=" * 60)
        logger.info("STEP 4: Normalizing features")
        logger.info("=" * 60)

        features_normalized, scaler = normalize_features(features_df, method='robust')

        # Step 5: Dimensionality reduction
        logger.info("=" * 60)
        logger.info("STEP 5: Reducing dimensionality")
        logger.info("=" * 60)

        # Get numeric features for reduction
        feature_cols = get_feature_names()
        feature_cols = [c for c in feature_cols if c in features_normalized.columns]

        features_array = features_normalized.select(feature_cols).to_numpy()

        # Remove rows with NaN
        valid_mask = ~np.isnan(features_array).any(axis=1)
        features_array_clean = features_array[valid_mask]
        logger.info(f"Removed {(~valid_mask).sum()} rows with NaN values")

        features_2d, reducers = reduce_dimensions(
            features_array_clean,
            n_components=2,
            method=dimensionality_method
        )

        results['steps']['dimensionality_reduction'] = {
            'method': dimensionality_method,
            'n_components': 2,
            'shape': features_2d.shape,
        }

        # Step 6: Clustering
        logger.info("=" * 60)
        logger.info("STEP 6: Clustering by geometric similarity")
        logger.info("=" * 60)

        cluster_result = cluster_by_geometry(
            features_array_clean,
            n_clusters=n_clusters,
            method=clustering_method
        )

        results['steps']['clustering'] = {
            'method': clustering_method,
            'n_clusters': cluster_result['n_clusters'],
            'silhouette_score': float(cluster_result['silhouette_score']),
            'davies_bouldin_score': float(cluster_result['davies_bouldin_score']),
        }

        # Step 7: Validation and relationship inference
        logger.info("=" * 60)
        logger.info("STEP 7: Validating clusters and inferring relationships")
        logger.info("=" * 60)

        # Add cluster labels to dataframe (only for valid rows)
        features_normalized_clean = features_normalized.filter(
            pl.lit(valid_mask)
        )
        features_with_clusters = features_normalized_clean.with_columns(
            pl.Series('cluster', cluster_result['labels'])
        )

        validation = validate_clusters(
            features_with_clusters,
            cluster_result['labels']
        )

        results['steps']['validation'] = {
            'cluster_profiles': {
                int(k): {str(k2): float(v2) if isinstance(v2, (int, float)) else str(v2)
                        for k2, v2 in v.items()}
                for k, v in validation['cluster_profiles'].items()
            },
            'relationship_signatures': {
                int(k): v.value
                for k, v in validation['relationship_signatures'].items()
            }
        }

        # Step 8: Visualizations
        logger.info("=" * 60)
        logger.info("STEP 8: Creating visualizations")
        logger.info("=" * 60)

        create_dashboard(
            features_with_clusters,
            features_2d,
            cluster_result['labels'],
            validation,
            self.output_dir
        )

        # Step 9: Save results
        logger.info("=" * 60)
        logger.info("STEP 9: Saving results")
        logger.info("=" * 60)

        self._save_results(features_with_clusters, results)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {self.output_dir}")

        return results

    def _load_and_sample(self, sample_size: int) -> pl.DataFrame:
        """Load and sample conversations from Parquet files."""
        # Find parquet files
        parquet_files = list(self.parquet_dir.glob('*.parquet'))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_dir}")

        logger.info(f"Found {len(parquet_files)} parquet file(s)")

        # Show stats for first file
        stats = get_parquet_stats(str(parquet_files[0]))
        logger.info(f"File stats: {stats}")

        # Load from first file for now (can extend to multiple later)
        df = load_conversations(
            str(parquet_files[0]),
            sample_size=sample_size,
            strategy='random',
            min_conversation_length=4
        )

        # Validate quality
        quality = validate_data_quality(df)
        logger.info(f"Data quality checks: {quality}")

        return df

    def _extract_features(self, pairs: List[Dict]) -> pl.DataFrame:
        """
        Build semantic surface and extract geometric features for all message pairs.

        This is the core feature extraction pipeline that transforms raw message
        pairs into geometric feature vectors for clustering and labeling.

        Pipeline flow:
        1. Load sentence transformer encoder
        2. Extract unique messages from all pairs
        3. Compute embeddings for unique messages (batch encoding for efficiency)
        4. Build semantic surface from embeddings
        5. For each pair: Extract full geometric feature suite
        6. Combine features with pair metadata (conversation_id, distance, type)
        7. Return as Polars DataFrame for downstream processing

        Feature categories extracted per pair:
        - Distance metrics (geodesic, Euclidean, cosine)
        - Path curvature (total, max, mean angular changes)
        - Displacement (vector magnitude and direction)
        - Influence overlap (Bernstein basis similarity)
        - Tangent alignment (surface flow direction similarity)
        - Topology (control point neighborhoods)

        Error handling:
        - Skips pairs that fail feature extraction
        - Logs errors but continues processing
        - Returns only successfully extracted features

        Args:
            pairs: List of pair dicts with 'msg1', 'msg2', 'conversation_id',
                   'distance', 'pair_type' keys

        Returns:
            Polars DataFrame with geometric features + metadata per pair
        """
        # Load encoder
        logger.info(f"Loading encoder: {self.encoder_model_name}")
        self.encoder = SentenceTransformer(self.encoder_model_name)

        # Extract unique messages
        unique_messages = list(set([p['msg1'] for p in pairs] + [p['msg2'] for p in pairs]))
        logger.info(f"Encoding {len(unique_messages)} unique messages")

        # Compute embeddings
        embeddings = self.encoder.encode(
            unique_messages,
            show_progress_bar=True,
            batch_size=32
        )

        # Build semantic surface
        logger.info("Building semantic surface")
        self.surface = SemanticSurface(
            messages=unique_messages,
            embeddings=embeddings,
            model_name=self.encoder_model_name
        )

        # Extract features
        logger.info("Extracting geometric features")
        extractor = GeometricFeatureExtractor(self.surface)

        # Build embedding lookup
        embedding_lookup = {msg: emb for msg, emb in zip(unique_messages, embeddings)}

        features_list = []
        for pair_dict in pairs:
            try:
                # Get embeddings for this pair
                emb1 = embedding_lookup.get(pair_dict['msg1'])
                emb2 = embedding_lookup.get(pair_dict['msg2'])

                features = extractor.extract_all_features(
                    pair_dict['msg1'],
                    pair_dict['msg2'],
                    embedding1=emb1,
                    embedding2=emb2
                )
                # Add metadata
                features['conversation_id'] = pair_dict['conversation_id']
                features['distance'] = pair_dict['distance']
                features['pair_type'] = pair_dict['pair_type']

                features_list.append(features)

            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                continue

        logger.info(f"Extracted features for {len(features_list)} pairs")

        return pl.DataFrame(features_list)

    def _save_results(self, features_df: pl.DataFrame, results: Dict):
        """Save labeled dataset and results."""
        # Save features as parquet
        output_parquet = self.output_dir / 'labeled_pairs.parquet'
        features_df.write_parquet(output_parquet)
        logger.info(f"Saved labeled pairs to {output_parquet}")

        # Save results as JSON
        output_json = self.output_dir / 'validation_report.json'
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved validation report to {output_json}")

        # Save summary
        summary = self._create_summary(results)
        output_summary = self.output_dir / 'summary.txt'
        with open(output_summary, 'w') as f:
            f.write(summary)
        logger.info(f"Saved summary to {output_summary}")

    def _create_summary(self, results: Dict) -> str:
        """
        Generate human-readable text summary of pipeline execution.

        Formats the complete pipeline results dict into a readable report
        with sections for configuration, data sampling, pair extraction,
        feature extraction, clustering, labeling, and validation metrics.

        Report structure:
        - Header with pipeline name
        - Configuration parameters used
        - Sampling statistics (messages, conversations)
        - Pair extraction stats (total pairs, strategy)
        - Feature extraction results (surface size, feature counts)
        - Clustering results (algorithm, number of clusters)
        - Label inference results (label distribution)
        - Validation metrics (purity, separation, quality indicators)
        - Feature importance rankings (if available)
        - Example pairs from each inferred cluster

        Formatting:
        - Uses ASCII box drawing for section headers
        - Indented hierarchical structure
        - Numeric values rounded for readability
        - Percentages for distributions

        Args:
            results: Complete results dict from pipeline.run() containing
                     config, steps, validation, and examples

        Returns:
            Formatted summary string ready to write to file or display
        """
        lines = []
        lines.append("=" * 70)
        lines.append("GEOMETRIC RELATIONSHIP INFERENCE - RESULTS SUMMARY")
        lines.append("=" * 70)
        lines.append("")

        # Config
        lines.append("Configuration:")
        for key, value in results['config'].items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Sampling
        sampling = results['steps']['sampling']
        lines.append(f"Sampled Data:")
        lines.append(f"  Total messages: {sampling['total_messages']}")
        lines.append(f"  Unique conversations: {sampling['unique_conversations']}")
        lines.append("")

        # Pairs
        pairs = results['steps']['pair_extraction']
        lines.append(f"Message Pairs:")
        lines.append(f"  Total pairs: {pairs['total_pairs']}")
        lines.append(f"  Strategy: {pairs['pair_strategy']}")
        lines.append("")

        # Features
        features = results['steps']['feature_extraction']
        lines.append(f"Feature Extraction:")
        lines.append(f"  Pairs with features: {features['n_pairs']}")
        lines.append(f"  Features per pair: {features['n_features']}")
        lines.append("")

        # Clustering
        clustering = results['steps']['clustering']
        lines.append(f"Clustering:")
        lines.append(f"  Method: {clustering['method']}")
        lines.append(f"  Clusters found: {clustering['n_clusters']}")
        lines.append(f"  Silhouette score: {clustering['silhouette_score']:.3f} (higher = better)")
        lines.append(f"  Davies-Bouldin: {clustering['davies_bouldin_score']:.3f} (lower = better)")
        lines.append("")

        # Relationship signatures
        signatures = results['steps']['validation']['relationship_signatures']
        profiles = results['steps']['validation']['cluster_profiles']

        lines.append("Inferred Relationship Types:")
        for cluster_id, rel_type in signatures.items():
            profile = profiles[cluster_id]
            count = profile['count']
            lines.append(f"  Cluster {cluster_id}: {rel_type.upper()} ({count} pairs)")

            # Show key metrics if available
            if 'median_normalized_curvature' in profile:
                lines.append(f"    Curvature: {profile['median_normalized_curvature']:.4f}")
            if 'median_cosine_similarity' in profile:
                lines.append(f"    Similarity: {profile['median_cosine_similarity']:.3f}")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"Output directory: {self.output_dir}")
        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    """Example usage of the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Geometric Relationship Inference Pipeline')
    parser.add_argument('parquet_dir', help='Directory containing parquet files')
    parser.add_argument('--output', '-o', default='./geometric_labels', help='Output directory')
    parser.add_argument('--sample-size', type=int, default=5000, help='Number of conversations to sample')
    parser.add_argument('--max-pairs', type=int, default=5000, help='Maximum message pairs')
    parser.add_argument('--clusters', type=int, default=5, help='Number of clusters')

    args = parser.parse_args()

    pipeline = GeometricLabelingPipeline(
        parquet_dir=args.parquet_dir,
        output_dir=args.output
    )

    results = pipeline.run_full_pipeline(
        sample_size=args.sample_size,
        max_pairs=args.max_pairs,
        n_clusters=args.clusters
    )

    print("\nPipeline completed successfully!")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
