"""
Geometric feature extraction for message pairs on semantic surfaces.

Extracts comprehensive geometric signatures including distances,
curvatures, and topological properties.
"""

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from typing import Tuple, Dict, List, Optional
import logging

# Import Luna Nine surface math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "luna9-python"))

from luna9.surface_math import (
    evaluate_surface,
    dS_du, dS_dv,
    geodesic_distance,
    compute_path_curvature,
    compute_curvature,
    semantic_displacement,
    project_to_surface
)
from luna9.semantic_surface import SemanticSurface

logger = logging.getLogger(__name__)


class GeometricFeatureExtractor:
    """
    Extract comprehensive geometric features from message pairs on semantic surfaces.

    Features include:
    - Distance metrics (geodesic, Euclidean, cosine, parameter space)
    - Path curvature (total, mean, max)
    - Surface topology (Gaussian and mean curvature at endpoints)
    - Semantic displacement (direction and magnitude)
    - Influence analysis (control point overlap)
    - Tangent vector alignment
    """

    def __init__(self, surface: Optional[SemanticSurface] = None):
        """
        Initialize feature extractor.

        Args:
            surface: Optional pre-built SemanticSurface. If None, must call set_surface()
        """
        self.surface = surface

    def set_surface(self, surface: SemanticSurface):
        """Set the semantic surface to use for feature extraction."""
        self.surface = surface

    def extract_all_features(
        self,
        msg1: str,
        msg2: str,
        embedding1: Optional[np.ndarray] = None,
        embedding2: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Extract all geometric features for a message pair.

        Args:
            msg1, msg2: The two messages
            embedding1, embedding2: Optional pre-computed embeddings

        Returns:
            Dict of feature name -> value
        """
        if self.surface is None:
            raise ValueError("Surface not set. Call set_surface() first.")

        # Get embeddings if not provided
        # Note: SemanticSurface doesn't expose encoder, so we need embeddings passed in
        # or we reconstruct from the surface's existing embeddings
        if embedding1 is None or embedding2 is None:
            # Try to find messages in surface's message list
            try:
                idx1 = self.surface.messages.index(msg1)
                idx2 = self.surface.messages.index(msg2)
                # Get embeddings from surface's control points
                # Control points are shaped (m, n, d) - flatten to get all embeddings
                all_embeddings = self.surface.control_points.reshape(-1, self.surface.control_points.shape[-1])
                embedding1 = all_embeddings[idx1] if embedding1 is None else embedding1
                embedding2 = all_embeddings[idx2] if embedding2 is None else embedding2
            except (ValueError, AttributeError):
                # If we can't find them, we'll get errors later - that's OK
                pass

        # Project to surface to get UV coordinates
        u1, v1, _ = project_to_surface(
            embedding1,
            self.surface.control_points,
            self.surface.weights
        )
        u2, v2, _ = project_to_surface(
            embedding2,
            self.surface.control_points,
            self.surface.weights
        )
        uv1 = np.array([u1, v1])
        uv2 = np.array([u2, v2])

        features = {
            'msg1': msg1,
            'msg2': msg2,
        }

        # Extract all feature groups
        features.update(self._extract_distance_features(uv1, uv2, embedding1, embedding2))
        features.update(self._extract_curvature_features(uv1, uv2))
        features.update(self._extract_topology_features(uv1, uv2))
        features.update(self._extract_displacement_features(uv1, uv2))
        features.update(self._extract_influence_features(uv1, uv2))
        features.update(self._extract_tangent_features(uv1, uv2))

        return features

    def _extract_distance_features(
        self,
        uv1: Tuple[float, float],
        uv2: Tuple[float, float],
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> Dict:
        """Extract distance metrics between message pair."""
        features = {}

        # Geodesic distance (arc length along surface)
        try:
            features['geodesic_distance'] = geodesic_distance(
                self.surface.control_points,
                self.surface.weights,
                tuple(uv1),
                tuple(uv2),
                num_steps=100
            )
        except Exception as e:
            logger.warning(f"Geodesic distance computation failed: {e}")
            features['geodesic_distance'] = np.nan

        # Euclidean distance in embedding space
        features['euclidean_distance'] = euclidean(embedding1, embedding2)

        # Cosine distance (1 - cosine similarity)
        features['cosine_distance'] = cosine(embedding1, embedding2)
        features['cosine_similarity'] = 1 - features['cosine_distance']

        # Parameter space distance
        features['parameter_distance'] = np.sqrt(
            (uv1[0] - uv2[0])**2 + (uv1[1] - uv2[1])**2
        )

        return features

    def _extract_curvature_features(
        self,
        uv1: Tuple[float, float],
        uv2: Tuple[float, float]
    ) -> Dict:
        """Extract path curvature features along geodesic."""
        features = {}

        try:
            curv_data = compute_path_curvature(
                self.surface.control_points,
                self.surface.weights,
                tuple(uv1),
                tuple(uv2),
                num_steps=100
            )

            features['total_path_curvature'] = curv_data['total_curvature']
            features['mean_path_curvature'] = curv_data['mean_curvature']
            features['max_path_curvature'] = curv_data['max_curvature']
            features['arc_length'] = curv_data['arc_length']

            # Normalized curvature (curvature per unit length)
            if features['arc_length'] > 1e-6:
                features['normalized_curvature'] = (
                    features['total_path_curvature'] / features['arc_length']
                )
            else:
                features['normalized_curvature'] = 0.0

        except Exception as e:
            logger.warning(f"Path curvature computation failed: {e}")
            features['total_path_curvature'] = np.nan
            features['mean_path_curvature'] = np.nan
            features['max_path_curvature'] = np.nan
            features['arc_length'] = np.nan
            features['normalized_curvature'] = np.nan

        return features

    def _extract_topology_features(
        self,
        uv1: Tuple[float, float],
        uv2: Tuple[float, float]
    ) -> Dict:
        """Extract surface topology features at endpoints."""
        features = {}

        try:
            # Gaussian and mean curvature at point 1
            K1, H1 = compute_curvature(
                self.surface.control_points,
                self.surface.weights,
                uv1[0], uv1[1]
            )
            features['gaussian_curvature_1'] = K1
            features['mean_curvature_1'] = H1

            # Gaussian and mean curvature at point 2
            K2, H2 = compute_curvature(
                self.surface.control_points,
                self.surface.weights,
                uv2[0], uv2[1]
            )
            features['gaussian_curvature_2'] = K2
            features['mean_curvature_2'] = H2

            # Curvature differences
            features['gaussian_curvature_diff'] = abs(K1 - K2)
            features['mean_curvature_diff'] = abs(H1 - H2)

            # Average curvatures
            features['avg_gaussian_curvature'] = (K1 + K2) / 2
            features['avg_mean_curvature'] = (H1 + H2) / 2

        except Exception as e:
            logger.warning(f"Topology feature computation failed: {e}")
            for key in ['gaussian_curvature_1', 'mean_curvature_1',
                       'gaussian_curvature_2', 'mean_curvature_2',
                       'gaussian_curvature_diff', 'mean_curvature_diff',
                       'avg_gaussian_curvature', 'avg_mean_curvature']:
                features[key] = np.nan

        return features

    def _extract_displacement_features(
        self,
        uv1: Tuple[float, float],
        uv2: Tuple[float, float]
    ) -> Dict:
        """Extract semantic displacement features."""
        features = {}

        try:
            disp_data = semantic_displacement(
                self.surface.control_points,
                self.surface.weights,
                tuple(uv1),
                tuple(uv2)
            )

            features['displacement_magnitude'] = disp_data['distance']

            # Direction vector - use first 3 components for interpretability
            direction = disp_data['direction']
            features['displacement_dir_x'] = direction[0] if len(direction) > 0 else 0
            features['displacement_dir_y'] = direction[1] if len(direction) > 1 else 0
            features['displacement_dir_z'] = direction[2] if len(direction) > 2 else 0

        except Exception as e:
            logger.warning(f"Displacement feature computation failed: {e}")
            features['displacement_magnitude'] = np.nan
            features['displacement_dir_x'] = np.nan
            features['displacement_dir_y'] = np.nan
            features['displacement_dir_z'] = np.nan

        return features

    def _extract_influence_features(
        self,
        uv1: Tuple[float, float],
        uv2: Tuple[float, float]
    ) -> Dict:
        """
        Compute Bernstein influence overlap metrics between two surface points.

        Analyzes how control points influence each location on the semantic
        surface using Bernstein basis functions. Points with high influence
        overlap share similar control point neighborhoods, suggesting semantic
        relatedness. Points with distinct influence patterns may represent
        opposed or distant semantic relationships.

        Computed metrics:
        - influence_overlap: Dot product of normalized influence weight vectors
          (1.0 = identical influence, 0.0 = orthogonal influence)
        - influence_entropy_1/2: Shannon entropy of influence distributions
          (high = diffuse influence, low = concentrated/peaked influence)
        - avg_influence_entropy: Mean entropy across both points
        - influence_concentration_1/2: Maximum influence weight per point
          (how dominated is each point by its strongest control point)

        Interpretation:
        - High overlap + low entropy: Both points strongly influenced by same
          control points → likely Direct or Related relationship
        - Low overlap + high entropy: Points draw from different control
          neighborhoods → likely Distant or Opposed relationship
        - High concentration: Point near a single control point (corner/edge)
        - Low concentration: Point in smooth interior region

        Args:
            uv1: (u,v) coordinates of first point
            uv2: (u,v) coordinates of second point

        Returns:
            Dict with influence_overlap, entropy, and concentration metrics
        """
        features = {}

        try:
            # Compute influence weights for each point
            influence1 = self.surface.compute_influence(uv1[0], uv1[1])
            influence2 = self.surface.compute_influence(uv2[0], uv2[1])

            # Extract weights into arrays
            weights1 = np.array([w for _, w in influence1])
            weights2 = np.array([w for _, w in influence2])

            # Normalize to sum to 1
            weights1 = weights1 / (weights1.sum() + 1e-10)
            weights2 = weights2 / (weights2.sum() + 1e-10)

            # Influence overlap (dot product of normalized weights)
            features['influence_overlap'] = np.dot(weights1, weights2)

            # Influence entropy (how spread out are influences?)
            entropy1 = -np.sum(weights1 * np.log(weights1 + 1e-10))
            entropy2 = -np.sum(weights2 * np.log(weights2 + 1e-10))
            features['influence_entropy_1'] = entropy1
            features['influence_entropy_2'] = entropy2
            features['avg_influence_entropy'] = (entropy1 + entropy2) / 2

            # Influence concentration (max weight)
            features['influence_concentration_1'] = weights1.max()
            features['influence_concentration_2'] = weights2.max()

        except Exception as e:
            logger.warning(f"Influence feature computation failed: {e}")
            features['influence_overlap'] = np.nan
            features['influence_entropy_1'] = np.nan
            features['influence_entropy_2'] = np.nan
            features['avg_influence_entropy'] = np.nan
            features['influence_concentration_1'] = np.nan
            features['influence_concentration_2'] = np.nan

        return features

    def _extract_tangent_features(
        self,
        uv1: Tuple[float, float],
        uv2: Tuple[float, float]
    ) -> Dict:
        """
        Measure tangent space alignment between two surface points.

        Computes first derivatives (∂S/∂u, ∂S/∂v) at both points and measures
        angular alignment between corresponding tangent vectors. Tangent alignment
        reveals whether the semantic surface is "flowing" in similar directions
        at both locations.

        Geometric intuition:
        - Tangent vectors define the local orientation of the surface
        - Aligned tangents suggest points lie on a common semantic "ridge"
        - Misaligned tangents suggest points lie in different semantic flows

        Computed metrics:
        - tangent_u_alignment: Angle between ∂S/∂u vectors (radians, 0 to π)
        - tangent_v_alignment: Angle between ∂S/∂v vectors (radians, 0 to π)
        - avg_tangent_alignment: Mean of u and v alignments

        Interpretation:
        - Small angles (near 0): Tangents point in same direction → similar
          semantic flow, likely Related or Direct relationship
        - Large angles (near π): Tangents oppose → different semantic directions,
          possibly Opposed or Distant relationship
        - Medium angles (≈π/2): Orthogonal tangent spaces → independently
          positioned concepts

        Args:
            uv1: (u,v) coordinates of first point
            uv2: (u,v) coordinates of second point

        Returns:
            Dict with tangent_u_alignment, tangent_v_alignment, avg metrics
        """
        features = {}

        try:
            # Tangent vectors at point 1
            Su1 = dS_du(self.surface.control_points, self.surface.weights, uv1[0], uv1[1])
            Sv1 = dS_dv(self.surface.control_points, self.surface.weights, uv1[0], uv1[1])

            # Tangent vectors at point 2
            Su2 = dS_du(self.surface.control_points, self.surface.weights, uv2[0], uv2[1])
            Sv2 = dS_dv(self.surface.control_points, self.surface.weights, uv2[0], uv2[1])

            # Normalize
            Su1_norm = Su1 / (np.linalg.norm(Su1) + 1e-10)
            Sv1_norm = Sv1 / (np.linalg.norm(Sv1) + 1e-10)
            Su2_norm = Su2 / (np.linalg.norm(Su2) + 1e-10)
            Sv2_norm = Sv2 / (np.linalg.norm(Sv2) + 1e-10)

            # Alignment angles (via dot product)
            features['tangent_u_alignment'] = float(np.arccos(
                np.clip(np.dot(Su1_norm, Su2_norm), -1, 1)
            ))
            features['tangent_v_alignment'] = float(np.arccos(
                np.clip(np.dot(Sv1_norm, Sv2_norm), -1, 1)
            ))

            # Average alignment
            features['avg_tangent_alignment'] = (
                features['tangent_u_alignment'] + features['tangent_v_alignment']
            ) / 2

        except Exception as e:
            logger.warning(f"Tangent feature computation failed: {e}")
            features['tangent_u_alignment'] = np.nan
            features['tangent_v_alignment'] = np.nan
            features['avg_tangent_alignment'] = np.nan

        return features

    def extract_batch_features(
        self,
        pairs: List[Tuple[str, str]],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Extract features for a batch of message pairs.

        Args:
            pairs: List of (msg1, msg2) tuples
            show_progress: Whether to show progress bar

        Returns:
            List of feature dictionaries
        """
        if self.surface is None:
            raise ValueError("Surface not set. Call set_surface() first.")

        features_list = []

        if show_progress:
            try:
                from tqdm import tqdm
                pairs = tqdm(pairs, desc="Extracting features")
            except ImportError:
                pass

        for msg1, msg2 in pairs:
            try:
                features = self.extract_all_features(msg1, msg2)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Feature extraction failed for pair: {e}")
                # Add empty features with NaNs
                features_list.append({
                    'msg1': msg1,
                    'msg2': msg2,
                    'error': str(e)
                })

        return features_list


def get_feature_names() -> List[str]:
    """
    Get list of all feature names extracted by GeometricFeatureExtractor.

    Useful for feature normalization and selection.
    """
    return [
        # Distance features
        'geodesic_distance',
        'euclidean_distance',
        'cosine_distance',
        'cosine_similarity',
        'parameter_distance',

        # Curvature features
        'total_path_curvature',
        'mean_path_curvature',
        'max_path_curvature',
        'arc_length',
        'normalized_curvature',

        # Topology features
        'gaussian_curvature_1',
        'mean_curvature_1',
        'gaussian_curvature_2',
        'mean_curvature_2',
        'gaussian_curvature_diff',
        'mean_curvature_diff',
        'avg_gaussian_curvature',
        'avg_mean_curvature',

        # Displacement features
        'displacement_magnitude',
        'displacement_dir_x',
        'displacement_dir_y',
        'displacement_dir_z',

        # Influence features
        'influence_overlap',
        'influence_entropy_1',
        'influence_entropy_2',
        'avg_influence_entropy',
        'influence_concentration_1',
        'influence_concentration_2',

        # Tangent features
        'tangent_u_alignment',
        'tangent_v_alignment',
        'avg_tangent_alignment',
    ]
