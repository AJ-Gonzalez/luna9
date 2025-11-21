"""
Hash bucketing for semantic surface coordinates.

Provides O(1) message retrieval by hashing surface (u,v) coordinates
into fixed-size buckets. Adapted from audio fingerprinting techniques.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pickle

import numpy as np


@dataclass
class MessageEntry:
    """Entry in the hash index."""
    message_id: int
    u: float
    v: float
    timestamp: datetime

    def distance_to(self, u: float, v: float) -> float:
        """Euclidean distance to query point."""
        return np.sqrt((self.u - u)**2 + (self.v - v)**2)


class HashIndex:
    """
    Surface-coordinate hash index for O(1) message retrieval.

    Messages are hashed based on their (u,v) coordinates on the semantic
    surface. Queries retrieve candidates from nearby buckets, then rank
    by geometric distance.

    Attributes:
        buckets: Dict mapping hash values to lists of MessageEntry
        bucket_size: Maximum entries per bucket (default 100)
        quantization_bits: Bits per coordinate (default 8 = 256 bins)
    """

    def __init__(self, bucket_size: int = 100, quantization_bits: int = 8):
        """
        Initialize hash index.

        Args:
            bucket_size: Maximum messages per bucket (older entries dropped when full)
            quantization_bits: Coordinate quantization resolution (8 bits = 256 levels)
        """
        self.buckets: Dict[int, List[MessageEntry]] = {}
        self.bucket_size = bucket_size
        self.quantization_bits = quantization_bits
        self.quantization_levels = 2 ** quantization_bits - 1  # e.g., 255 for 8 bits

    def _quantize(self, value: float) -> int:
        """
        Quantize coordinate from [0,1] to discrete bin.

        Args:
            value: Coordinate in [0,1]

        Returns:
            Quantized value in [0, quantization_levels]
        """
        clamped = np.clip(value, 0.0, 1.0)
        return int(clamped * self.quantization_levels)

    def _compute_hash(self, u: float, v: float) -> int:
        """
        Compute 32-bit hash from surface coordinates.

        Hash layout (for 8-bit quantization):
            Bits 0-7:   u coordinate (quantized)
            Bits 8-15:  v coordinate (quantized)
            Bits 16-31: Reserved for future use (coordinate differences, etc.)

        Args:
            u: Surface u coordinate [0,1]
            v: Surface v coordinate [0,1]

        Returns:
            32-bit hash value
        """
        u_q = self._quantize(u)
        v_q = self._quantize(v)

        # Simple hash: pack u and v into lower 16 bits
        # Upper 16 bits reserved for coordinate differences (future enhancement)
        return (v_q << 8) | u_q

    def add_message(
        self,
        message_id: int,
        u: float,
        v: float,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Add message to index.

        Args:
            message_id: Unique message identifier
            u: Surface u coordinate [0,1]
            v: Surface v coordinate [0,1]
            timestamp: Message timestamp (defaults to now)

        Returns:
            Hash value where message was stored
        """
        if timestamp is None:
            timestamp = datetime.now()

        hash_val = self._compute_hash(u, v)
        entry = MessageEntry(message_id, u, v, timestamp)

        # Create bucket if doesn't exist
        if hash_val not in self.buckets:
            self.buckets[hash_val] = []

        bucket = self.buckets[hash_val]
        bucket.append(entry)

        # Enforce bucket size limit (FIFO eviction)
        if len(bucket) > self.bucket_size:
            bucket.pop(0)  # Remove oldest entry

        return hash_val

    def _get_neighbor_hashes(self, u: float, v: float, radius: int = 1) -> List[int]:
        """
        Get hash values for neighboring quantization bins.

        Args:
            u: Query u coordinate
            v: Query v coordinate
            radius: Search radius in quantization bins (default 1 = 9 neighbors)

        Returns:
            List of hash values to check
        """
        u_q = self._quantize(u)
        v_q = self._quantize(v)

        hashes = []
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                u_neighbor = np.clip(u_q + du, 0, self.quantization_levels)
                v_neighbor = np.clip(v_q + dv, 0, self.quantization_levels)

                # Reconstruct hash from neighboring coordinates
                hash_val = (v_neighbor << 8) | u_neighbor
                hashes.append(hash_val)

        return hashes

    def query(
        self,
        u: float,
        v: float,
        k: int = 10,
        search_radius: int = 1
    ) -> List[MessageEntry]:
        """
        Find k nearest messages to query point.

        Args:
            u: Query u coordinate [0,1]
            v: Query v coordinate [0,1]
            k: Number of results to return
            search_radius: Quantization bin radius to search (1 = check 9 bins)

        Returns:
            List of up to k MessageEntry objects, sorted by distance
        """
        # Get candidate hashes
        candidate_hashes = self._get_neighbor_hashes(u, v, search_radius)

        # Collect all candidates from matching buckets
        candidates: List[MessageEntry] = []
        for hash_val in candidate_hashes:
            if hash_val in self.buckets:
                candidates.extend(self.buckets[hash_val])

        # Sort by distance to query point
        candidates.sort(key=lambda entry: entry.distance_to(u, v))

        # Return top k
        return candidates[:k]

    def get_initial_guess(self, surface, embedding: np.ndarray) -> Tuple[float, float]:
        """
        Get initial (u, v) guess for projection using nearest control point.

        This provides a warm start for Newton-Raphson projection by finding
        the control point closest to the query embedding in semantic space,
        then returning its (u, v) coordinates.

        Args:
            surface: SemanticSurface object (needed for control points)
            embedding: Query embedding to project

        Returns:
            (u_init, v_init) tuple for projection starting point
        """
        # Find nearest control point in embedding space (brute force, but only once)
        min_dist = float('inf')
        best_i, best_j = 0, 0

        for i in range(surface.grid_m):
            for j in range(surface.grid_n):
                cp = surface.control_points[i, j]
                dist = np.linalg.norm(embedding - cp)
                if dist < min_dist:
                    min_dist = dist
                    best_i, best_j = i, j

        # Convert grid indices to (u, v) coordinates
        u_init = best_i / max(1, surface.grid_m - 1)
        v_init = best_j / max(1, surface.grid_n - 1)

        return u_init, v_init

    def stats(self) -> Dict:
        """
        Get index statistics.

        Returns:
            Dict with bucket count, total messages, avg bucket size, etc.
        """
        total_messages = sum(len(bucket) for bucket in self.buckets.values())
        num_buckets = len(self.buckets)
        avg_bucket_size = total_messages / num_buckets if num_buckets > 0 else 0
        max_bucket_size = max((len(b) for b in self.buckets.values()), default=0)

        return {
            'num_buckets': num_buckets,
            'total_messages': total_messages,
            'avg_bucket_size': avg_bucket_size,
            'max_bucket_size': max_bucket_size,
            'bucket_capacity': self.bucket_size,
            'quantization_bits': self.quantization_bits,
            'quantization_levels': self.quantization_levels
        }

    def save(self, path: Path) -> None:
        """
        Serialize index to disk.

        Args:
            path: File path for serialized index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'buckets': self.buckets,
                'bucket_size': self.bucket_size,
                'quantization_bits': self.quantization_bits
            }, f)

    @classmethod
    def load(cls, path: Path) -> 'HashIndex':
        """
        Deserialize index from disk.

        Args:
            path: File path to serialized index

        Returns:
            HashIndex instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = cls(
            bucket_size=data['bucket_size'],
            quantization_bits=data['quantization_bits']
        )
        index.buckets = data['buckets']

        return index
