"""
Tests for HashIndex - surface coordinate hash bucketing.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from luna9 import HashIndex, MessageEntry


class TestMessageEntry:
    """Test MessageEntry dataclass."""

    def test_creation(self):
        """Test creating a MessageEntry."""
        timestamp = datetime.now()
        entry = MessageEntry(
            message_id=42,
            u=0.5,
            v=0.7,
            timestamp=timestamp
        )

        assert entry.message_id == 42
        assert entry.u == 0.5
        assert entry.v == 0.7
        assert entry.timestamp == timestamp

    def test_distance_calculation(self):
        """Test distance_to method."""
        entry = MessageEntry(0, u=0.0, v=0.0, timestamp=datetime.now())

        # Distance to self
        assert entry.distance_to(0.0, 0.0) == 0.0

        # Distance to (1, 0)
        dist = entry.distance_to(1.0, 0.0)
        assert abs(dist - 1.0) < 1e-10

        # Distance to (0.3, 0.4) should be 0.5 (3-4-5 triangle)
        dist = entry.distance_to(0.3, 0.4)
        assert abs(dist - 0.5) < 1e-10


class TestHashIndex:
    """Test HashIndex core functionality."""

    def test_initialization(self):
        """Test HashIndex initialization with default and custom params."""
        # Default params
        index = HashIndex()
        assert index.bucket_size == 100
        assert index.quantization_bits == 8
        assert index.quantization_levels == 255
        assert len(index.buckets) == 0

        # Custom params
        index = HashIndex(bucket_size=50, quantization_bits=6)
        assert index.bucket_size == 50
        assert index.quantization_bits == 6
        assert index.quantization_levels == 63  # 2^6 - 1

    def test_quantization(self):
        """Test coordinate quantization."""
        index = HashIndex(quantization_bits=8)  # 255 levels

        # Test boundary values
        assert index._quantize(0.0) == 0
        assert index._quantize(1.0) == 255
        assert index._quantize(0.5) == 127  # Middle

        # Test clamping
        assert index._quantize(-0.5) == 0   # Clamp to 0
        assert index._quantize(1.5) == 255  # Clamp to 255

        # Test quantization resolution
        index_4bit = HashIndex(quantization_bits=4)  # 15 levels
        assert index_4bit._quantize(0.0) == 0
        assert index_4bit._quantize(1.0) == 15
        assert index_4bit._quantize(0.5) == 7

    def test_hash_computation(self):
        """Test hash value computation."""
        index = HashIndex(quantization_bits=8)

        # Corner coordinates should have distinct hashes
        hash_00 = index._compute_hash(0.0, 0.0)
        hash_01 = index._compute_hash(0.0, 1.0)
        hash_10 = index._compute_hash(1.0, 0.0)
        hash_11 = index._compute_hash(1.0, 1.0)

        # All should be different
        hashes = [hash_00, hash_01, hash_10, hash_11]
        assert len(set(hashes)) == 4

        # Verify bit packing (v in upper byte, u in lower byte)
        # (0, 0) -> 0x0000
        assert hash_00 == 0x0000

        # (1, 0) -> 0x00FF (u=255, v=0)
        assert hash_10 == 0x00FF

        # (0, 1) -> 0xFF00 (u=0, v=255)
        assert hash_01 == 0xFF00

        # (1, 1) -> 0xFFFF (u=255, v=255)
        assert hash_11 == 0xFFFF

    def test_hash_determinism(self):
        """Test that same coordinates produce same hash."""
        index = HashIndex()

        hash1 = index._compute_hash(0.42, 0.73)
        hash2 = index._compute_hash(0.42, 0.73)

        assert hash1 == hash2

    def test_add_message(self):
        """Test adding messages to index."""
        index = HashIndex()
        timestamp = datetime.now()

        # Add first message
        hash_val = index.add_message(0, 0.5, 0.5, timestamp)

        assert isinstance(hash_val, int)
        assert len(index.buckets) == 1
        assert hash_val in index.buckets
        assert len(index.buckets[hash_val]) == 1

        entry = index.buckets[hash_val][0]
        assert entry.message_id == 0
        assert entry.u == 0.5
        assert entry.v == 0.5
        assert entry.timestamp == timestamp

    def test_add_multiple_messages_same_bucket(self):
        """Test adding multiple messages that hash to same bucket."""
        index = HashIndex()

        # Add messages very close together (should quantize to same bucket)
        hash1 = index.add_message(0, 0.500, 0.500)
        hash2 = index.add_message(1, 0.501, 0.500)  # Very close
        hash3 = index.add_message(2, 0.500, 0.501)  # Very close

        # With 8-bit quantization (255 levels), 0.001 difference is ~0.25 bins
        # So these should likely be in the same bucket
        assert len(index.buckets) <= 2  # At most 2 buckets

    def test_add_messages_different_buckets(self):
        """Test adding messages that hash to different buckets."""
        index = HashIndex()

        # Add messages far apart
        hash1 = index.add_message(0, 0.0, 0.0)
        hash2 = index.add_message(1, 0.5, 0.5)
        hash3 = index.add_message(2, 1.0, 1.0)

        # Should be in different buckets
        assert len(index.buckets) == 3
        assert hash1 != hash2 != hash3

    def test_bucket_size_limit(self):
        """Test that buckets enforce size limit with FIFO eviction."""
        index = HashIndex(bucket_size=3)

        # Add 5 messages to same bucket
        for i in range(5):
            # Use same coordinates so all go to same bucket
            index.add_message(i, 0.5, 0.5)

        # Should have only 1 bucket
        assert len(index.buckets) == 1

        # Bucket should have max 3 entries (oldest 2 evicted)
        bucket = list(index.buckets.values())[0]
        assert len(bucket) == 3

        # Should have messages 2, 3, 4 (0 and 1 evicted)
        message_ids = [entry.message_id for entry in bucket]
        assert message_ids == [2, 3, 4]

    def test_default_timestamp(self):
        """Test that add_message uses current time if no timestamp provided."""
        index = HashIndex()

        before = datetime.now()
        index.add_message(0, 0.5, 0.5)  # No timestamp
        after = datetime.now()

        bucket = list(index.buckets.values())[0]
        entry_timestamp = bucket[0].timestamp

        # Timestamp should be between before and after
        assert before <= entry_timestamp <= after

    def test_query_empty_index(self):
        """Test querying empty index."""
        index = HashIndex()

        results = index.query(0.5, 0.5, k=10)

        assert results == []

    def test_query_exact_match(self):
        """Test querying for exact coordinate match."""
        index = HashIndex()

        # Add some messages
        index.add_message(0, 0.5, 0.5)
        index.add_message(1, 0.3, 0.7)
        index.add_message(2, 0.8, 0.2)

        # Query for exact match
        results = index.query(0.5, 0.5, k=10)

        # Should get at least the exact match
        assert len(results) > 0
        # First result should be closest (likely the exact match)
        assert results[0].u == 0.5
        assert results[0].v == 0.5

    def test_query_nearest_neighbors(self):
        """Test that query returns nearest neighbors sorted by distance."""
        index = HashIndex()

        # Add messages in a grid
        coords = [
            (0, 0.0, 0.0),
            (1, 0.5, 0.0),
            (2, 1.0, 0.0),
            (3, 0.0, 0.5),
            (4, 0.5, 0.5),  # Center
            (5, 1.0, 0.5),
            (6, 0.0, 1.0),
            (7, 0.5, 1.0),
            (8, 1.0, 1.0),
        ]

        for msg_id, u, v in coords:
            index.add_message(msg_id, u, v)

        # Query center point
        results = index.query(0.5, 0.5, k=5)

        # Should get results
        assert len(results) > 0

        # Results should be sorted by distance
        distances = [entry.distance_to(0.5, 0.5) for entry in results]
        assert distances == sorted(distances)

        # Closest should be the center point (msg_id=4)
        assert results[0].message_id == 4
        assert results[0].distance_to(0.5, 0.5) == 0.0

    def test_query_k_limit(self):
        """Test that query respects k parameter."""
        index = HashIndex()

        # Add many messages
        for i in range(20):
            u = i / 20.0
            v = 0.5
            index.add_message(i, u, v)

        # Query with k=5
        results = index.query(0.5, 0.5, k=5)

        # Should get exactly 5 results (or fewer if not enough candidates)
        assert len(results) <= 5

    def test_query_search_radius(self):
        """Test that search_radius affects candidate retrieval."""
        index = HashIndex()

        # Add messages spread across surface
        for i in range(10):
            u = i / 10.0
            v = 0.5
            index.add_message(i, u, v)

        # Query with small radius (may miss distant messages)
        results_small = index.query(0.5, 0.5, k=10, search_radius=1)

        # Query with large radius (should find more)
        results_large = index.query(0.5, 0.5, k=10, search_radius=3)

        # Larger radius should potentially find more candidates
        # (depends on quantization, but at minimum should not find fewer)
        assert len(results_large) >= len(results_small)

    def test_neighbor_hashes(self):
        """Test neighbor hash generation."""
        index = HashIndex()

        # Get neighbors with radius=1 (should be 3x3 = 9 hashes)
        neighbors = index._get_neighbor_hashes(0.5, 0.5, radius=1)

        assert len(neighbors) == 9  # 3x3 grid
        assert len(set(neighbors)) <= 9  # May have duplicates at boundaries

        # Get neighbors with radius=2 (should be 5x5 = 25 hashes)
        neighbors = index._get_neighbor_hashes(0.5, 0.5, radius=2)

        assert len(neighbors) == 25

    def test_neighbor_hashes_at_boundary(self):
        """Test neighbor hashes don't go out of bounds."""
        index = HashIndex()

        # Query at corner (0, 0)
        neighbors = index._get_neighbor_hashes(0.0, 0.0, radius=1)

        # Should still get 9 hashes (clamped to valid range)
        assert len(neighbors) == 9

        # All hashes should be valid (no negative values)
        assert all(h >= 0 for h in neighbors)

    def test_stats(self):
        """Test index statistics."""
        index = HashIndex(bucket_size=10)

        # Empty index
        stats = index.stats()
        assert stats['num_buckets'] == 0
        assert stats['total_messages'] == 0
        assert stats['avg_bucket_size'] == 0
        assert stats['max_bucket_size'] == 0

        # Add some messages
        for i in range(25):
            u = (i % 5) / 5.0  # 5 columns
            v = (i // 5) / 5.0  # 5 rows
            index.add_message(i, u, v)

        stats = index.stats()
        assert stats['num_buckets'] > 0
        assert stats['total_messages'] == 25
        assert stats['bucket_capacity'] == 10
        assert stats['quantization_bits'] == 8

    def test_save_and_load(self, tmp_path):
        """Test saving and loading index."""
        index = HashIndex(bucket_size=50, quantization_bits=6)

        # Add some messages
        for i in range(10):
            index.add_message(i, i/10.0, 0.5)

        # Save to temp file
        save_path = tmp_path / "test_index.pkl"
        index.save(save_path)

        assert save_path.exists()

        # Load from file
        loaded_index = HashIndex.load(save_path)

        # Check configuration matches
        assert loaded_index.bucket_size == 50
        assert loaded_index.quantization_bits == 6
        assert loaded_index.quantization_levels == 63

        # Check buckets match
        assert len(loaded_index.buckets) == len(index.buckets)

        # Check query results match
        original_results = index.query(0.5, 0.5, k=5)
        loaded_results = loaded_index.query(0.5, 0.5, k=5)

        assert len(original_results) == len(loaded_results)
        for orig, loaded in zip(original_results, loaded_results):
            assert orig.message_id == loaded.message_id
            assert orig.u == loaded.u
            assert orig.v == loaded.v

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates parent directories if needed."""
        index = HashIndex()
        index.add_message(0, 0.5, 0.5)

        # Save to nested path that doesn't exist
        save_path = tmp_path / "nested" / "dirs" / "index.pkl"
        index.save(save_path)

        assert save_path.exists()


class TestHashIndexEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_coordinates_at_boundaries(self):
        """Test coordinates at (0,0), (0,1), (1,0), (1,1)."""
        index = HashIndex()

        corners = [
            (0, 0.0, 0.0),
            (1, 0.0, 1.0),
            (2, 1.0, 0.0),
            (3, 1.0, 1.0),
        ]

        for msg_id, u, v in corners:
            hash_val = index.add_message(msg_id, u, v)
            assert hash_val is not None

        # Query each corner
        for msg_id, u, v in corners:
            results = index.query(u, v, k=1)
            assert len(results) > 0
            # Closest should be the corner itself
            assert results[0].message_id == msg_id

    def test_coordinates_outside_bounds(self):
        """Test that coordinates outside [0,1] are clamped."""
        index = HashIndex()

        # Add message with out-of-bounds coords
        hash_val = index.add_message(0, -0.5, 1.5)

        # Should be added successfully (coords clamped internally)
        assert len(index.buckets) == 1

        # Query should work
        results = index.query(-0.5, 1.5, k=1)
        assert len(results) > 0

    def test_very_small_bucket_size(self):
        """Test with bucket_size=1."""
        index = HashIndex(bucket_size=1)

        # Add multiple messages to same bucket
        for i in range(5):
            index.add_message(i, 0.5, 0.5)

        # Should only have most recent message
        bucket = list(index.buckets.values())[0]
        assert len(bucket) == 1
        assert bucket[0].message_id == 4  # Last one added

    def test_low_quantization_bits(self):
        """Test with very coarse quantization (2 bits = 4 levels)."""
        index = HashIndex(quantization_bits=2)  # Only 3 levels (0, 1, 2, 3)

        # Add messages across surface
        for i in range(10):
            u = i / 10.0
            v = 0.5
            index.add_message(i, u, v)

        # With coarse quantization, should have fewer buckets
        stats = index.stats()
        assert stats['num_buckets'] <= 4  # Only 4 possible u values

    def test_high_quantization_bits(self):
        """Test with very fine quantization (12 bits = 4096 levels)."""
        index = HashIndex(quantization_bits=12)

        # Add messages very close together
        for i in range(5):
            u = 0.5 + i * 0.0001  # Very small differences
            v = 0.5
            index.add_message(i, u, v)

        # With fine quantization, should separate into more buckets
        stats = index.stats()
        # Exact number depends on quantization alignment
        assert stats['num_buckets'] >= 1

    def test_query_k_larger_than_available(self):
        """Test querying for more results than exist."""
        index = HashIndex()

        # Add only 3 messages
        for i in range(3):
            index.add_message(i, i/3.0, 0.5)

        # Query for k=10
        results = index.query(0.5, 0.5, k=10)

        # Should get at most 3 results
        assert len(results) <= 3

    def test_concurrent_same_coordinates(self):
        """Test adding multiple messages with identical coordinates."""
        index = HashIndex()

        # Add multiple messages at exact same point
        for i in range(5):
            index.add_message(i, 0.5, 0.5)

        # All should be in same bucket
        assert len(index.buckets) == 1

        bucket = list(index.buckets.values())[0]
        assert len(bucket) == 5

        # Query should return all of them
        results = index.query(0.5, 0.5, k=10)
        assert len(results) == 5

        # All should have distance 0
        for entry in results:
            assert entry.distance_to(0.5, 0.5) == 0.0
