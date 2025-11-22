"""
Geometric relationship inference for Luna Nine.

This module provides tools to infer semantic relationship types
(necessary, opposed, ancillary, hierarchical, irrelevant) from
geometric properties of message pairs on semantic surfaces.

The core hypothesis: semantic relationships have distinct geometric signatures
that can be detected without manual labeling or LLM inference.
"""

__version__ = "0.1.0"

from .parquet_utils import load_conversations, sample_pairs
from .geometry_extractor import GeometricFeatureExtractor
from .clustering import cluster_by_geometry, validate_clusters
from .pipeline import GeometricLabelingPipeline

__all__ = [
    'load_conversations',
    'sample_pairs',
    'GeometricFeatureExtractor',
    'cluster_by_geometry',
    'validate_clusters',
    'GeometricLabelingPipeline',
]
