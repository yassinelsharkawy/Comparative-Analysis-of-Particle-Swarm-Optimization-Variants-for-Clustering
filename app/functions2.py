import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import math

# Set global random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define a helper fitness function that can be used internally
def _clustering_fitness(X, k, pos):
    """Calculates fitness (sum of squared distances) for a given position (flattened centroids)."""
    n_samples, n_features = X.shape
    centers = pos.reshape(k, n_features)
    # Ensure centers are valid (not NaN or Inf) before calculating distances
    if not np.all(np.isfinite(centers)):
        return np.inf # Return infinite fitness for invalid positions

    # Calculate distances and find closest center for each sample
    # Note: pairwise_distances_argmin_min returns (indices, distances)
    _, dists = pairwise_distances_argmin_min(X, centers)
    return np.sum(dists**2)

# Helper function to get labels and centroids from best position
def _get_labels_and_centroids(X, k, best_pos):
    """Reshapes best_pos to centroids and assigns labels to data points."""
    n_samples, n_features = X.shape
    best_centers = best_pos.reshape(k, n_features)
    # Assign each data point in X to the closest centroid
    labels, _ = pairwise_distances_argmin_min(X, best_centers)
    return labels, best_centers


def pso_global(X, k, swarm_size=30, max_iters=100, w=0.5, c1=1.5, c2=1.5):
    """Global-best PSO for clustering."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Feature-wise bounds for initialization
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)

    # Initialize swarm positions and velocities
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))

    # Personal bests
    pbest_pos = positions.copy()
    # Use the helper fitness function
    pbest_fit = np.array([_clustering_fitness(X, k, p) for p in positions])


    # Global best
    # Handle case where all initial fitness values are inf (unlikely but safe)
    if np.all(np.isinf(pbest_fit)):
         gbest_idx = 0 # Or handle as an error state
    else:
        gbest_idx = np.argmin(pbest_fit)

    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    # PSO main loop
    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))
        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)

        # Use the helper fitness function
        fitness_vals = np.array([_clustering_fitness(X, k, p) for p in positions])

        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]

        # Handle case where all current pbest_fit values are inf
        if not np.all(np.isinf(pbest_fit)):
            min_idx = np.argmin(pbest_fit)
            if pbest_fit[min_idx] < gbest_fit:
                gbest_fit = pbest_fit[min_idx]
                gbest_pos = pbest_pos[min_idx].copy()

    # --- FIX: Calculate and return labels and centroids from the best position ---
    labels, best_centers = _get_labels_and_centroids(X, k, gbest_pos)
    return labels, best_centers


def pso_local(X, k, swarm_size=30, max_iters=100, w=0.5, c1=1.5, c2=1.5):
    """Local-best PSO for clustering with ring neighborhood."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Feature-wise bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)

    # Initialize positions & velocities
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))

    # Personal bests
    pbest_pos = positions.copy()
    # Use the helper fitness function
    pbest_fit = np.array([_clustering_fitness(X, k, p) for p in positions])

    # Local best positions (initially same as personal)
    lbest_pos = pbest_pos.copy()
    lbest_fit = pbest_fit.copy()

    # Update local best for each particle's ring neighborhood
    def update_local_bests():
        for i in range(swarm_size):
            neighbors = [(i-1) % swarm_size, i, (i+1) % swarm_size]
            # Handle case where all neighbor pbest_fit values are inf
            neighbor_fits = pbest_fit[neighbors]
            if np.all(np.isinf(neighbor_fits)):
                 best_idx_in_neighbors = 1 # default to particle itself if neighbors are inf
            else:
                 best_idx_in_neighbors = np.argmin(neighbor_fits)

            best_neighbor_global_idx = neighbors[best_idx_in_neighbors]
            lbest_pos[i] = pbest_pos[best_neighbor_global_idx]
            lbest_fit[i] = pbest_fit[best_neighbor_global_idx]


    update_local_bests()

    # PSO main loop
    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (lbest_pos - positions))
        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)

        # Use the helper fitness function
        fitness_vals = np.array([_clustering_fitness(X, k, p) for p in positions])

        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]
        update_local_bests()

    # Choose best among all personal bests (to get the best overall solution)
    # Handle case where all pbest_fit values are inf
    if np.all(np.isinf(pbest_fit)):
        best_pos_overall = positions[0] # Return initial position or handle as error
    else:
        best_idx = np.argmin(pbest_fit)
        best_pos_overall = pbest_pos[best_idx]


    # --- FIX: Calculate and return labels and centroids from the best position ---
    labels, best_centers = _get_labels_and_centroids(X, k, best_pos_overall)
    return labels, best_centers


def pso_linear_inertia(X, k, swarm_size=30, max_iters=100, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):
    """PSO with linearly decreasing inertia weight."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)

    # Initialize
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))
    pbest_pos = positions.copy()
    # Use the helper fitness function
    pbest_fit = np.array([_clustering_fitness(X, k, p) for p in positions])

    # Global best init
    if np.all(np.isinf(pbest_fit)):
         gbest_idx = 0
    else:
        gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    # Main loop
    for iter in range(max_iters):
        # Update inertia
        w = w_max - (w_max - w_min) * (iter / (max_iters - 1))
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))
        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)

        # Use the helper fitness function
        fitness_vals = np.array([_clustering_fitness(X, k, p) for p in positions])

        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]

        if not np.all(np.isinf(pbest_fit)):
            min_idx = np.argmin(pbest_fit)
            if pbest_fit[min_idx] < gbest_fit:
                gbest_fit = pbest_fit[min_idx]
                gbest_pos = pbest_pos[min_idx].copy()

    # --- FIX: Calculate and return labels and centroids from the best position ---
    labels, best_centers = _get_labels_and_centroids(X, k, gbest_pos)
    return labels, best_centers


def pso_constriction(X, k, swarm_size=30, max_iters=100, c1=2.05, c2=2.05):
    """PSO with constriction factor (Clerc's constriction coefficient)."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)

    # Calculate constriction coefficient
    phi = c1 + c2
    # Add check for phi <= 4 to avoid issues with sqrt
    if phi <= 4:
        # Handle potential division by zero or complex numbers if phi is not suitable
         print(f"Warning: phi ({phi}) <= 4 in constriction factor calculation. Using standard PSO update.")
         chi = 1.0 # Fallback to no constriction if formula is invalid
    else:
        denominator = abs(2 - phi - math.sqrt(phi**2 - 4*phi))
        if denominator == 0:
             print("Warning: Denominator is zero in constriction factor calculation. Using standard PSO update.")
             chi = 1.0 # Fallback
        else:
             chi = 2 / denominator


    # Initialize
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))
    pbest_pos = positions.copy()

    # Use the helper fitness function
    pbest_fit = np.array([_clustering_fitness(X, k, p) for p in positions])

    # Global best
    if np.all(np.isinf(pbest_fit)):
         gbest_idx = 0
    else:
        gbest_idx = np.argmin(pbest_fit)

    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    # PSO loop
    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = chi * (velocities
                            + c1 * r1 * (pbest_pos - positions)
                            + c2 * r2 * (gbest_pos - positions))
        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)

        # Use the helper fitness function
        fitness_vals = np.array([_clustering_fitness(X, k, p) for p in positions])

        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]

        if not np.all(np.isinf(pbest_fit)):
            min_idx = np.argmin(pbest_fit)
            if pbest_fit[min_idx] < gbest_fit:
                gbest_fit = pbest_fit[min_idx]
                gbest_pos = pbest_pos[min_idx].copy()

    # --- FIX: Calculate and return labels and centroids from the best position ---
    labels, best_centers = _get_labels_and_centroids(X, k, gbest_pos)
    return labels, best_centers


def pso_velocity_clamped(X, k, swarm_size=30, max_iters=100, w=0.5, c1=1.5, c2=1.5, vmax_frac=0.2):
    """PSO with velocity clamping."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)

    # Max velocity per dimension (based on the range of the data)
    data_range = data_max - data_min
    vmax = np.tile(data_range * vmax_frac, k) # Clamp velocity based on feature range

    # Initialize
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    # Initialize velocities within the clamped range
    velocities = np.random.uniform(-vmax, vmax, (swarm_size, dim))
    pbest_pos = positions.copy()

    # Use the helper fitness function
    pbest_fit = np.array([_clustering_fitness(X, k, p) for p in positions])

    if np.all(np.isinf(pbest_fit)):
         gbest_idx = 0
    else:
        gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))

        # Clamp velocities
        velocities = np.clip(velocities, -vmax, vmax) # Use the correct vmax array

        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)

        # Use the helper fitness function
        fitness_vals = np.array([_clustering_fitness(X, k, p) for p in positions])

        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]

        if not np.all(np.isinf(pbest_fit)):
            min_idx = np.argmin(pbest_fit)
            if pbest_fit[min_idx] < gbest_fit:
                gbest_fit = pbest_fit[min_idx]
                gbest_pos = pbest_pos[min_idx].copy()

    # --- FIX: Calculate and return labels and centroids from the best position ---
    labels, best_centers = _get_labels_and_centroids(X, k, gbest_pos)
    return labels, best_centers


def pso_mutation(X, k, swarm_size=30, max_iters=100, w=0.5, c1=1.5, c2=1.5, mutation_prob=0.2, mutation_scale=0.05):
    """PSO with Gaussian mutation injected into particles."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)
    data_range = data_max - data_min # Needed for mutation scale

    # Initialize
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))
    pbest_pos = positions.copy()

    # Use the helper fitness function
    pbest_fit = np.array([_clustering_fitness(X, k, p) for p in positions])

    if np.all(np.isinf(pbest_fit)):
         gbest_idx = 0
    else:
        gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))
        positions += velocities

        # Inject Gaussian mutation into a subset of particles
        for i in range(swarm_size):
            if np.random.rand() < mutation_prob:
                # Scale mutation based on data range per feature dimension
                mutation = np.random.normal(0, mutation_scale * np.tile(data_range, k), dim)
                positions[i] += mutation

        positions = np.clip(positions, pos_min, pos_max)

        # Use the helper fitness function
        fitness_vals = np.array([_clustering_fitness(X, k, p) for p in positions])

        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]

        if not np.all(np.isinf(pbest_fit)):
            min_idx = np.argmin(pbest_fit)
            if pbest_fit[min_idx] < gbest_fit:
                gbest_fit = pbest_fit[min_idx]
                gbest_pos = pbest_pos[min_idx].copy()

    # --- FIX: Calculate and return labels and centroids from the best position ---
    labels, best_centers = _get_labels_and_centroids(X, k, gbest_pos)
    return labels, best_centers

# Note: The other imports and helper functions like _clustering_fitness
# and _get_labels_and_centroids were added/defined at the top for clarity and reuse.