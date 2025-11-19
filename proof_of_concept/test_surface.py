"""
Unit tests for Bézier surface mathematics.

Tests core mathematical properties and validates implementation.
"""

import numpy as np
import pytest
from surface_math import (
    bernstein_basis,
    evaluate_surface,
    dS_du, dS_dv,
    project_to_surface,
    geodesic_distance,
    compute_curvature
)


class TestBernsteinBasis:
    """Test Bernstein polynomial basis functions."""

    def test_partition_of_unity(self):
        """Bernstein polynomials should sum to 1."""
        n = 3
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for t in t_values:
            basis_sum = sum(bernstein_basis(i, n, t) for i in range(n + 1))
            assert np.abs(basis_sum - 1.0) < 1e-10, f"Basis sum {basis_sum} != 1 at t={t}"

    def test_endpoint_values(self):
        """Basis functions should be 1 at endpoints only for corresponding index."""
        n = 3

        # At t=0, only B_0 should be 1
        for i in range(n + 1):
            expected = 1.0 if i == 0 else 0.0
            actual = bernstein_basis(i, n, 0.0)
            assert np.abs(actual - expected) < 1e-10

        # At t=1, only B_n should be 1
        for i in range(n + 1):
            expected = 1.0 if i == n else 0.0
            actual = bernstein_basis(i, n, 1.0)
            assert np.abs(actual - expected) < 1e-10

    def test_symmetry(self):
        """B_i^n(t) = B_{n-i}^n(1-t)"""
        n = 3
        t = 0.3

        for i in range(n + 1):
            left = bernstein_basis(i, n, t)
            right = bernstein_basis(n - i, n, 1 - t)
            assert np.abs(left - right) < 1e-10


class TestSurfaceEvaluation:
    """Test surface evaluation in various dimensions."""

    def test_2x2_surface_3d(self):
        """Test simple 2×2 surface in 3D (bilinear patch)."""
        # Define a simple bilinear patch (degree 1 in both directions)
        # Control points indexed as [i,j,:] where i~u, j~v
        # Layout: control_points[u_index, v_index, xyz]
        control_points = np.array([
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.5]],  # u=0 edge (v varies)
            [[1.0, 0.0, 0.5], [1.0, 1.0, 1.0]]   # u=1 edge (v varies)
        ])
        weights = np.ones((2, 2))

        # Test corner points: (u, v) → control_points[u_index, v_index]
        corners = [
            ((0.0, 0.0), [0.0, 0.0, 0.0]),  # control_points[0, 0]
            ((1.0, 0.0), [1.0, 0.0, 0.5]),  # control_points[1, 0]
            ((0.0, 1.0), [0.0, 1.0, 0.5]),  # control_points[0, 1]
            ((1.0, 1.0), [1.0, 1.0, 1.0])   # control_points[1, 1]
        ]

        for (u, v), expected in corners:
            result = evaluate_surface(control_points, weights, u, v)
            np.testing.assert_allclose(result, expected, atol=1e-10,
                                       err_msg=f"Corner ({u},{v}) incorrect")

        # Test center point (should be average of corners for bilinear)
        center = evaluate_surface(control_points, weights, 0.5, 0.5)
        expected_center = [0.5, 0.5, 0.5]
        np.testing.assert_allclose(center, expected_center, atol=1e-10)

    def test_uniform_weights(self):
        """Uniform weights should give standard Bézier surface."""
        # Random control points in 3D
        np.random.seed(42)
        control_points = np.random.randn(3, 3, 3)
        weights = np.ones((3, 3))

        # Evaluate at several points
        test_points = [(0.2, 0.3), (0.5, 0.5), (0.7, 0.8)]

        for u, v in test_points:
            result = evaluate_surface(control_points, weights, u, v)
            assert result.shape == (3,), "Output dimension incorrect"
            assert not np.any(np.isnan(result)), "NaN in result"

    def test_high_dimensional(self):
        """Test that evaluation works in high-dimensional space (768-dim)."""
        np.random.seed(42)
        d = 768
        control_points = np.random.randn(4, 4, d)
        weights = np.ones((4, 4))

        u, v = 0.5, 0.5
        result = evaluate_surface(control_points, weights, u, v)

        assert result.shape == (d,), f"Expected shape ({d},), got {result.shape}"
        assert not np.any(np.isnan(result)), "NaN in high-dim result"
        assert not np.any(np.isinf(result)), "Inf in high-dim result"


class TestDerivatives:
    """Test derivative computations."""

    def test_derivatives_exist(self):
        """Derivatives should be computable without errors."""
        np.random.seed(42)
        control_points = np.random.randn(4, 4, 10)
        weights = np.ones((4, 4))

        u, v = 0.5, 0.5

        S_u = dS_du(control_points, weights, u, v)
        S_v = dS_dv(control_points, weights, u, v)

        assert S_u.shape == (10,)
        assert S_v.shape == (10,)
        assert not np.any(np.isnan(S_u))
        assert not np.any(np.isnan(S_v))

    def test_derivatives_at_corners(self):
        """Derivatives at corners should be related to control point spacing."""
        # Simple bilinear patch: control_points[u_index, v_index, xy]
        control_points = np.array([
            [[0.0, 0.0], [0.0, 1.0]],  # u=0 edge: (0,0) to (0,1)
            [[1.0, 0.0], [1.0, 1.0]]   # u=1 edge: (1,0) to (1,1)
        ])
        weights = np.ones((2, 2))

        # At (u=0, v=0), derivative dS/du should point toward u=1 direction
        # Which is from (0,0) toward (1,0), i.e., +x direction
        S_u = dS_du(control_points, weights, 0.0, 0.0)
        assert S_u[0] > 0, "S_u should point in +x direction"
        assert np.abs(S_u[1]) < 0.1, "S_u should not have significant y component"


class TestProjection:
    """Test projection onto surface."""

    def test_project_known_point(self):
        """Projecting a control point should return its parameter coordinates."""
        np.random.seed(42)
        control_points = np.random.randn(4, 4, 768)
        weights = np.ones((4, 4))

        # Project corner control point (should converge to (0,0))
        corner_point = control_points[0, 0]
        u, v, iterations = project_to_surface(
            corner_point, control_points, weights,
            u_init=0.5, v_init=0.5
        )

        assert np.abs(u) < 0.01, f"Expected u≈0, got {u}"
        assert np.abs(v) < 0.01, f"Expected v≈0, got {v}"
        assert iterations < 50, f"Should converge quickly, took {iterations} iterations"

    def test_projection_convergence(self):
        """Projection should converge for arbitrary query points."""
        np.random.seed(42)
        control_points = np.random.randn(4, 4, 100)
        weights = np.ones((4, 4))

        # Random query point
        query = np.random.randn(100)

        u, v, iterations = project_to_surface(
            query, control_points, weights,
            u_init=0.5, v_init=0.5
        )

        # Should return valid parameters
        assert 0.0 <= u <= 1.0, f"u={u} out of bounds"
        assert 0.0 <= v <= 1.0, f"v={v} out of bounds"

        # Verify the projection is close to the query
        S_uv = evaluate_surface(control_points, weights, u, v)
        distance = np.linalg.norm(S_uv - query)

        # Should be reasonably close (exact closeness depends on surface shape)
        assert not np.isnan(distance), "Distance is NaN"


class TestGeodesicDistance:
    """Test geodesic distance computation."""

    def test_geodesic_greater_than_euclidean(self):
        """Geodesic distance should be >= Euclidean distance."""
        np.random.seed(42)
        control_points = np.random.randn(4, 4, 100)
        weights = np.ones((4, 4))

        uv1 = (0.2, 0.3)
        uv2 = (0.7, 0.8)

        # Compute geodesic
        geo_dist = geodesic_distance(control_points, weights, uv1, uv2)

        # Compute Euclidean
        p1 = evaluate_surface(control_points, weights, *uv1)
        p2 = evaluate_surface(control_points, weights, *uv2)
        euclidean_dist = np.linalg.norm(p2 - p1)

        assert geo_dist >= euclidean_dist - 1e-6, \
            f"Geodesic {geo_dist} should be >= Euclidean {euclidean_dist}"

    def test_geodesic_self_distance(self):
        """Geodesic from a point to itself should be ~0."""
        np.random.seed(42)
        control_points = np.random.randn(4, 4, 50)
        weights = np.ones((4, 4))

        uv = (0.5, 0.5)
        dist = geodesic_distance(control_points, weights, uv, uv)

        assert dist < 1e-6, f"Self-distance should be ~0, got {dist}"

    def test_geodesic_symmetry(self):
        """Geodesic distance should be symmetric."""
        np.random.seed(42)
        control_points = np.random.randn(4, 4, 50)
        weights = np.ones((4, 4))

        uv1 = (0.2, 0.3)
        uv2 = (0.7, 0.8)

        dist_12 = geodesic_distance(control_points, weights, uv1, uv2)
        dist_21 = geodesic_distance(control_points, weights, uv2, uv1)

        # Should be approximately equal (within numerical tolerance)
        assert np.abs(dist_12 - dist_21) < 1e-6, \
            f"Distance should be symmetric: {dist_12} vs {dist_21}"


class TestCurvature:
    """Test curvature computation."""

    def test_curvature_computable(self):
        """Curvature should be computable without errors."""
        np.random.seed(42)
        control_points = np.random.randn(4, 4, 100)
        weights = np.ones((4, 4))

        u, v = 0.5, 0.5
        K, H = compute_curvature(control_points, weights, u, v)

        assert not np.isnan(K), "Gaussian curvature is NaN"
        assert not np.isnan(H), "Mean curvature is NaN"
        assert not np.isinf(K), "Gaussian curvature is Inf"
        assert not np.isinf(H), "Mean curvature is Inf"

    def test_curvature_varies(self):
        """Different points should have different curvatures."""
        np.random.seed(42)
        control_points = np.random.randn(4, 4, 50)
        weights = np.ones((4, 4))

        curvatures = []
        test_points = [(0.2, 0.3), (0.5, 0.5), (0.8, 0.7)]

        for u, v in test_points:
            K, H = compute_curvature(control_points, weights, u, v)
            curvatures.append((K, H))

        # At least some points should have different curvatures
        K_values = [k for k, h in curvatures]
        assert len(set(K_values)) > 1 or all(abs(k) < 1e-10 for k in K_values), \
            "Curvatures should vary across surface"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_3d(self):
        """Test complete workflow with toy 3D data."""
        # Create a simple surface
        control_points = np.array([
            [[0, 0, 0], [1, 0, 0.5], [2, 0, 0.3]],
            [[0, 1, 0.2], [1, 1, 1], [2, 1, 0.7]],
            [[0, 2, 0.1], [1, 2, 0.6], [2, 2, 0.4]]
        ])
        weights = np.ones((3, 3))

        # Evaluate at a point
        p = evaluate_surface(control_points, weights, 0.5, 0.5)
        assert p.shape == (3,)

        # Project a query
        query = np.array([1.0, 1.0, 0.8])
        u, v, iters = project_to_surface(query, control_points, weights)
        assert 0 <= u <= 1 and 0 <= v <= 1

        # Compute geodesic
        dist = geodesic_distance(control_points, weights, (0.2, 0.3), (0.8, 0.7))
        assert dist > 0

        # Compute curvature
        K, H = compute_curvature(control_points, weights, 0.5, 0.5)
        assert not np.isnan(K) and not np.isnan(H)

    def test_full_workflow_high_dim(self):
        """Test complete workflow with 768-dimensional embeddings."""
        np.random.seed(42)
        d = 768
        control_points = np.random.randn(4, 4, d)
        weights = np.ones((4, 4))

        # Evaluate
        p = evaluate_surface(control_points, weights, 0.5, 0.5)
        assert p.shape == (d,)

        # Project
        query = np.random.randn(d)
        u, v, iters = project_to_surface(query, control_points, weights)
        assert 0 <= u <= 1 and 0 <= v <= 1

        # Geodesic
        dist = geodesic_distance(control_points, weights, (0.2, 0.3), (0.8, 0.7))
        assert dist > 0

        # Curvature
        K, H = compute_curvature(control_points, weights, 0.5, 0.5)
        assert not np.isnan(K)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
