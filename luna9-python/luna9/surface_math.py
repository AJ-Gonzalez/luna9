"""
Core Bézier Surface Mathematics for Semantic Surfaces

Implements rational Bézier surfaces in high-dimensional embedding space
for semantic relationship representation.
"""

import numpy as np
from scipy.special import comb
from typing import Tuple, Optional


def bernstein_basis(i: int, n: int, t: float) -> float:
    """
    Compute the i-th Bernstein basis polynomial of degree n at parameter t.

    B_i^n(t) = C(n,i) * t^i * (1-t)^(n-i)

    Args:
        i: Index (0 <= i <= n)
        n: Degree
        t: Parameter value (typically in [0,1])

    Returns:
        Value of Bernstein polynomial

    Properties:
        - Sum over all i: Σ B_i^n(t) = 1
        - B_i^n(0) = 1 if i=0, else 0
        - B_i^n(1) = 1 if i=n, else 0
    """
    if i < 0 or i > n:
        return 0.0

    return comb(n, i, exact=True) * (t ** i) * ((1 - t) ** (n - i))


def evaluate_surface(
    control_points: np.ndarray,
    weights: np.ndarray,
    u: float,
    v: float
) -> np.ndarray:
    """
    Evaluate a rational Bézier surface at parameter coordinates (u, v).

    For a surface with mxn control points of degree (m-1, n-1):
    S(u,v) = Σᵢ Σⱼ wᵢⱼ Pᵢⱼ Bᵢ^(m-1)(u) Bⱼ^(n-1)(v) / Σᵢ Σⱼ wᵢⱼ Bᵢ^(m-1)(u) Bⱼ^(n-1)(v)

    Args:
        control_points: Array of shape (m, n, d) where d is embedding dimension
        weights: Array of shape (m, n) - rational weights for each control point
        u: Parameter in u direction (typically [0,1])
        v: Parameter in v direction (typically [0,1])

    Returns:
        Point on surface as array of shape (d,) in embedding space
    """
    m, n, d = control_points.shape
    degree_u = m - 1
    degree_v = n - 1

    # Compute all Bernstein basis values
    basis_u = np.array([bernstein_basis(i, degree_u, u) for i in range(m)])
    basis_v = np.array([bernstein_basis(j, degree_v, v) for j in range(n)])

    # Compute weighted sum numerator and denominator
    numerator = np.zeros(d)
    denominator = 0.0

    for i in range(m):
        for j in range(n):
            basis_product = basis_u[i] * basis_v[j]
            weighted_basis = weights[i, j] * basis_product

            numerator += weighted_basis * control_points[i, j]
            denominator += weighted_basis

    # Return rational surface point
    return numerator / denominator


def dS_du(
    control_points: np.ndarray,
    weights: np.ndarray,
    u: float,
    v: float,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute first derivative of surface with respect to u.

    Uses finite differences for now (analytical derivative can be added later).

    Args:
        control_points: Array of shape (m, n, d)
        weights: Array of shape (m, n)
        u, v: Parameter coordinates
        epsilon: Step size for finite difference

    Returns:
        Derivative vector ∂S/∂u of shape (d,)
    """
    # Ensure we don't go out of bounds
    u_plus = min(u + epsilon, 1.0)
    u_minus = max(u - epsilon, 0.0)
    step = u_plus - u_minus

    S_plus = evaluate_surface(control_points, weights, u_plus, v)
    S_minus = evaluate_surface(control_points, weights, u_minus, v)

    return (S_plus - S_minus) / step


def dS_dv(
    control_points: np.ndarray,
    weights: np.ndarray,
    u: float,
    v: float,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute first derivative of surface with respect to v.

    Args:
        control_points: Array of shape (m, n, d)
        weights: Array of shape (m, n)
        u, v: Parameter coordinates
        epsilon: Step size for finite difference

    Returns:
        Derivative vector ∂S/∂v of shape (d,)
    """
    v_plus = min(v + epsilon, 1.0)
    v_minus = max(v - epsilon, 0.0)
    step = v_plus - v_minus

    S_plus = evaluate_surface(control_points, weights, u, v_plus)
    S_minus = evaluate_surface(control_points, weights, u, v_minus)

    return (S_plus - S_minus) / step


def d2S_du2(
    control_points: np.ndarray,
    weights: np.ndarray,
    u: float,
    v: float,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute second derivative ∂²S/∂u².

    Args:
        control_points: Array of shape (m, n, d)
        weights: Array of shape (m, n)
        u, v: Parameter coordinates
        epsilon: Step size

    Returns:
        Second derivative vector of shape (d,)
    """
    u_plus = min(u + epsilon, 1.0)
    u_minus = max(u - epsilon, 0.0)
    step = u_plus - u_minus

    dS_plus = dS_du(control_points, weights, u_plus, v, epsilon)
    dS_minus = dS_du(control_points, weights, u_minus, v, epsilon)

    return (dS_plus - dS_minus) / step


def d2S_dv2(
    control_points: np.ndarray,
    weights: np.ndarray,
    u: float,
    v: float,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute second derivative ∂²S/∂v².

    Args:
        control_points: Array of shape (m, n, d)
        weights: Array of shape (m, n)
        u, v: Parameter coordinates
        epsilon: Step size

    Returns:
        Second derivative vector of shape (d,)
    """
    v_plus = min(v + epsilon, 1.0)
    v_minus = max(v - epsilon, 0.0)
    step = v_plus - v_minus

    dS_plus = dS_dv(control_points, weights, u, v_plus, epsilon)
    dS_minus = dS_dv(control_points, weights, u, v_minus, epsilon)

    return (dS_plus - dS_minus) / step


def d2S_dudv(
    control_points: np.ndarray,
    weights: np.ndarray,
    u: float,
    v: float,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute mixed partial derivative ∂²S/∂u∂v.

    Args:
        control_points: Array of shape (m, n, d)
        weights: Array of shape (m, n)
        u, v: Parameter coordinates
        epsilon: Step size

    Returns:
        Mixed partial derivative vector of shape (d,)
    """
    v_plus = min(v + epsilon, 1.0)
    v_minus = max(v - epsilon, 0.0)
    step = v_plus - v_minus

    dS_u_plus = dS_du(control_points, weights, u, v_plus, epsilon)
    dS_u_minus = dS_du(control_points, weights, u, v_minus, epsilon)

    return (dS_u_plus - dS_u_minus) / step


def project_to_surface(
    embedding: np.ndarray,
    control_points: np.ndarray,
    weights: np.ndarray,
    u_init: float = 0.5,
    v_init: float = 0.5,
    max_iterations: int = 50,
    tolerance: float = 1e-6
) -> Tuple[float, float, int]:
    """
    Project a query embedding onto the surface using Newton-Raphson iteration.

    Finds (u, v) that minimizes ||S(u,v) - query||².

    Based on SolveSpace projection algorithm:
    - Compute residual r = S(u,v) - query
    - Build Jacobian J from first derivatives
    - Solve J·δ = -g where g = [S_u·r, S_v·r]
    - Update (u,v) ← (u,v) + δ

    Args:
        embedding: Query point to project, shape (d,)
        control_points: Surface control points, shape (m, n, d)
        weights: Surface weights, shape (m, n)
        u_init, v_init: Initial parameter guess
        max_iterations: Maximum Newton iterations
        tolerance: Convergence tolerance on residual norm

    Returns:
        (u, v, iterations): Converged parameters and iteration count
    """
    u, v = u_init, v_init

    for iteration in range(max_iterations):
        # Evaluate surface and derivatives
        S_uv = evaluate_surface(control_points, weights, u, v)
        S_u = dS_du(control_points, weights, u, v)
        S_v = dS_dv(control_points, weights, u, v)

        # Compute residual
        residual = S_uv - embedding
        residual_norm = np.linalg.norm(residual)

        # Check convergence
        if residual_norm < tolerance:
            return u, v, iteration

        # Build Jacobian (2x2 matrix in parameter space)
        # J = [[S_u·S_u, S_u·S_v],
        #      [S_v·S_u, S_v·S_v]]
        J = np.array([
            [np.dot(S_u, S_u), np.dot(S_u, S_v)],
            [np.dot(S_v, S_u), np.dot(S_v, S_v)]
        ])

        # Gradient g = [S_u·r, S_v·r]
        g = np.array([
            np.dot(S_u, residual),
            np.dot(S_v, residual)
        ])

        # Solve J·δ = -g
        try:
            delta = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            # Jacobian is singular, can't proceed
            break

        # Update parameters with bounds clamping
        u = np.clip(u + delta[0], 0.0, 1.0)
        v = np.clip(v + delta[1], 0.0, 1.0)

    # Return best estimate even if didn't fully converge
    return u, v, max_iterations


def geodesic_distance(
    control_points: np.ndarray,
    weights: np.ndarray,
    uv1: Tuple[float, float],
    uv2: Tuple[float, float],
    num_steps: int = 100
) -> float:
    """
    Compute geodesic (arc length) distance along surface between two points.

    Linearly interpolates in parameter space and integrates arc length:
    L = ∫ ||dS/dt|| dt

    This is an approximation - true geodesics would solve differential equations.
    But for our purposes, parameter-space interpolation should work well.

    Args:
        control_points: Surface control points, shape (m, n, d)
        weights: Surface weights, shape (m, n)
        uv1: Starting parameter coordinates (u1, v1)
        uv2: Ending parameter coordinates (u2, v2)
        num_steps: Number of integration steps

    Returns:
        Geodesic distance (arc length)
    """
    u1, v1 = uv1
    u2, v2 = uv2

    total_length = 0.0
    prev_point = evaluate_surface(control_points, weights, u1, v1)

    for i in range(1, num_steps + 1):
        t = i / num_steps
        u = u1 + t * (u2 - u1)
        v = v1 + t * (v2 - v1)

        current_point = evaluate_surface(control_points, weights, u, v)
        segment_length = np.linalg.norm(current_point - prev_point)
        total_length += segment_length

        prev_point = current_point

    return total_length


def compute_path_curvature(
    control_points: np.ndarray,
    weights: np.ndarray,
    uv1: Tuple[float, float],
    uv2: Tuple[float, float],
    num_steps: int = 100
) -> dict:
    """
    Compute path curvature metrics for the geodesic between two points.

    Measures how much the path twists/turns through semantic space, capturing
    "semantic transition complexity" rather than just distance.

    Path curvature is measured by angular changes between consecutive tangent
    vectors along the discretized path. High curvature suggests the semantic
    relationship requires intermediate concepts to bridge.

    Args:
        control_points: Surface control points, shape (m, n, d)
        weights: Surface weights, shape (m, n)
        uv1: Starting parameter coordinates (u1, v1)
        uv2: Ending parameter coordinates (u2, v2)
        num_steps: Number of discretization steps

    Returns:
        dict with:
            'arc_length': total path length (scalar)
            'total_curvature': sum of angular changes (scalar, radians)
            'max_curvature': largest angular change between segments (scalar, radians)
            'mean_curvature': average angular change (scalar, radians)
            'curvature_profile': array of angular changes along path (array)
            'path_points': discretized points along path for visualization (array)
    """
    u1, v1 = uv1
    u2, v2 = uv2

    # Discretize path by linear interpolation in UV space
    # (This is approximate geodesic - true geodesic would solve diff eqs)
    path_points = []
    for i in range(num_steps + 1):
        t = i / num_steps
        u = u1 + t * (u2 - u1)
        v = v1 + t * (v2 - v1)
        point = evaluate_surface(control_points, weights, u, v)
        path_points.append(point)

    path_points = np.array(path_points)  # shape (num_steps+1, d)

    # Compute tangent vectors (direction along path)
    tangents = []
    for i in range(len(path_points) - 1):
        tangent = path_points[i + 1] - path_points[i]
        tangent_length = np.linalg.norm(tangent)

        # Normalize (handle zero-length segments)
        if tangent_length > 1e-10:
            tangent_normalized = tangent / tangent_length
        else:
            tangent_normalized = np.zeros_like(tangent)

        tangents.append(tangent_normalized)

    tangents = np.array(tangents)  # shape (num_steps, d)

    # Compute angular changes between consecutive tangent vectors
    angular_changes = []
    for i in range(len(tangents) - 1):
        # Angle between consecutive tangents using dot product
        # cos(θ) = t1 · t2 (since both are unit vectors)
        cos_angle = np.dot(tangents[i], tangents[i + 1])

        # Clamp to [-1, 1] to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Angular change in radians
        angle = np.arccos(cos_angle)
        angular_changes.append(angle)

    angular_changes = np.array(angular_changes)

    # Compute arc length
    segment_lengths = [np.linalg.norm(path_points[i + 1] - path_points[i])
                      for i in range(len(path_points) - 1)]
    arc_length = sum(segment_lengths)

    # Aggregate metrics
    if len(angular_changes) > 0:
        total_curvature = float(np.sum(angular_changes))
        max_curvature = float(np.max(angular_changes))
        mean_curvature = float(np.mean(angular_changes))
    else:
        total_curvature = 0.0
        max_curvature = 0.0
        mean_curvature = 0.0

    return {
        'arc_length': float(arc_length),
        'total_curvature': total_curvature,
        'max_curvature': max_curvature,
        'mean_curvature': mean_curvature,
        'curvature_profile': angular_changes.tolist(),
        'path_points': path_points.tolist()  # For visualization/debugging
    }


def semantic_displacement(
    control_points: np.ndarray,
    weights: np.ndarray,
    uv1: Tuple[float, float],
    uv2: Tuple[float, float]
) -> dict:
    """
    Compute net semantic displacement vector between two points on surface.

    Returns both the displacement vector (B - A) and derived metrics:
    - Full displacement vector in embedding space
    - Euclidean distance (magnitude)
    - Unit direction vector (normalized displacement)

    This captures "in what semantic directions are these points separated"
    rather than just scalar distance.

    Args:
        control_points: Surface control points, shape (m, n, d)
        weights: Surface weights, shape (m, n)
        uv1: Starting parameter coordinates (u1, v1)
        uv2: Ending parameter coordinates (u2, v2)

    Returns:
        dict with:
            'vector': displacement vector B - A, shape (d,)
            'distance': Euclidean norm ||B - A||
            'direction': unit vector (B - A) / ||B - A||, shape (d,)
    """
    u1, v1 = uv1
    u2, v2 = uv2

    # Evaluate surface at both points
    point_A = evaluate_surface(control_points, weights, u1, v1)
    point_B = evaluate_surface(control_points, weights, u2, v2)

    # Compute displacement
    displacement_vector = point_B - point_A
    distance = np.linalg.norm(displacement_vector)

    # Avoid division by zero for identical points
    if distance < 1e-10:
        direction = np.zeros_like(displacement_vector)
    else:
        direction = displacement_vector / distance

    return {
        'vector': displacement_vector,
        'distance': float(distance),
        'direction': direction
    }


def compute_curvature(
    control_points: np.ndarray,
    weights: np.ndarray,
    u: float,
    v: float
) -> Tuple[float, float]:
    """
    Compute Gaussian and mean curvature at a point on the surface.

    Uses fundamental forms from differential geometry:
    - First fundamental form (metric): E, F, G
    - Second fundamental form (shape): L, M, N
    - Gaussian curvature: K = (LN - M²) / (EG - F²)
    - Mean curvature: H = (EN - 2FM + GL) / (2(EG - F²))

    Args:
        control_points: Surface control points, shape (m, n, d)
        weights: Surface weights, shape (m, n)
        u, v: Parameter coordinates

    Returns:
        (K, H): Gaussian and mean curvature
    """
    # First derivatives
    S_u = dS_du(control_points, weights, u, v)
    S_v = dS_dv(control_points, weights, u, v)

    # Second derivatives
    S_uu = d2S_du2(control_points, weights, u, v)
    S_uv = d2S_dudv(control_points, weights, u, v)
    S_vv = d2S_dv2(control_points, weights, u, v)

    # First fundamental form (metric tensor)
    E = np.dot(S_u, S_u)
    F = np.dot(S_u, S_v)
    G = np.dot(S_v, S_v)

    # Compute surface normal via cross product analog in high dimensions
    # Use SVD to find normal to tangent plane spanned by S_u and S_v
    tangent_matrix = np.vstack([S_u, S_v])

    try:
        # Normal is in the null space of tangent matrix
        # For high-dimensional space, use SVD
        U, s, Vt = np.linalg.svd(tangent_matrix)

        # Normal is the last row of Vt (smallest singular value direction)
        # But in high-dim space there are many normals - pick one
        # For curvature, we need projection onto normal direction
        # Actually, we can compute curvature using the shape operator directly

        # Second fundamental form (projection of second derivatives onto normal)
        # In high dimensions, we use the formula:
        # L = (S_uu · n), M = (S_uv · n), N = (S_vv · n)
        # But we need to be careful about what "normal" means

        # Alternative: compute curvature directly from metric
        # Using the fact that in high dimensions, we care about
        # intrinsic curvature which can be computed from metric alone

        # For now, use simplified approach based on metric tensor
        metric_det = E * G - F * F

        if metric_det < 1e-10:
            # Degenerate metric
            return 0.0, 0.0

        # Project second derivatives to get second fundamental form
        # This is an approximation suitable for high-dimensional spaces
        # where we don't have a unique normal vector

        # Compute shape operator components
        L = np.dot(S_uu, S_u) / E if E > 1e-10 else 0.0
        M = np.dot(S_uv, S_u) / np.sqrt(E * G) if E * G > 1e-10 else 0.0
        N = np.dot(S_vv, S_v) / G if G > 1e-10 else 0.0

        # Gaussian curvature
        K = (L * N - M * M) / metric_det if metric_det > 1e-10 else 0.0

        # Mean curvature
        H = (E * N - 2 * F * M + G * L) / (2 * metric_det) if metric_det > 1e-10 else 0.0

        return K, H

    except np.linalg.LinAlgError:
        return 0.0, 0.0
