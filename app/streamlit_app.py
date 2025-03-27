import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import os
import sys

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from functions2 import (pso_global, pso_local, pso_linear_inertia, 
                       pso_constriction, pso_velocity_clamped, pso_mutation)

# Set page title and layout
st.set_page_config(page_title="PSO Clustering Variations", layout="wide")

# App title and description
st.title("Particle Swarm Optimization Clustering Variations")
st.markdown("""
This project demonstrates different variations of Particle Swarm Optimization (PSO) for clustering.
You can adjust parameters for each PSO variation and see how it affects clustering results.
""")

# Sidebar for file upload and global settings
with st.sidebar:
    st.header("Data Settings")
    
    # Option to use default dataset or upload new one
    use_default = st.checkbox("Use default Mall Customer dataset", value=True)
    
    if use_default:
        # Use the default dataset path
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "Mall_Customers.csv")
        file_path = default_path
        
        # Display dataset info
        st.info("Using Mall Customer Dataset (Age, Annual Income, Spending Score)")
    else:
        # File uploader
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file is not None:
            file_path = uploaded_file
        else:
            st.warning("Please upload a CSV file")
            file_path = None
    
    # Global parameter settings
    st.header("Global Settings")
    
    # Number of clusters
    k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=5)
    
    # Features selection (will be shown only if we have the dataset)
    if 'df' in locals():
        features = st.multiselect("Features to use for clustering", 
                                  options=df.select_dtypes(include=np.number).columns.tolist(),
                                  default=df.select_dtypes(include=np.number).columns.tolist()[:3])
    
    # Random state
    random_state = st.slider("Random seed", min_value=0, max_value=100, value=42)
    np.random.seed(random_state)

# Function to load and preprocess the dataset
@st.cache_data
def load_data(file_path):
    try:
        if isinstance(file_path, str):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data if file_path is available
if file_path:
    df = load_data(file_path)
    
    if df is not None:
        # Display data overview
        with st.expander("Data Overview"):
            st.write("First few rows of the dataset:")
            st.dataframe(df.head())
            
            st.write("Dataset shape:", df.shape)
            
            st.write("Descriptive statistics:")
            st.dataframe(df.describe())
        
        # Get numeric features if not already selected
        if 'features' not in locals():
            features = df.select_dtypes(include=np.number).columns.tolist()
            # Exclude CustomerID if present
            if 'CustomerID' in features:
                features.remove('CustomerID')
            
            # Update sidebar with features
            with st.sidebar:
                features = st.multiselect("Features to use for clustering", 
                                         options=df.select_dtypes(include=np.number).columns.tolist(),
                                         default=features[:3])
        
        # Preprocess data
        @st.cache_data
        def preprocess_data(df, features):
            X = df[features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            return X, X_scaled, scaler
        
        X, X_scaled, scaler = preprocess_data(df, features)
        
        # Data visualization
        with st.expander("Data Visualization"):
            if len(features) >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X[:, 0], X[:, 1], alpha=0.7)
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_title("Feature Space Visualization")
                st.pyplot(fig)
                
                # Correlation matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title("Correlation Matrix")
                st.pyplot(fig)
                
                # PCA if more than 2 features
                if len(features) > 2:
                    pca = PCA(n_components=2, random_state=random_state)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title("PCA Projection (2 Components)")
                    st.pyplot(fig)
                    
                    # Explained variance
                    st.write(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
                      # Create tabs for each PSO variation
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "K-Means (Baseline)", 
            "Global-Best PSO", 
            "Local-Best PSO", 
            "Linear Inertia PSO", 
            "Constriction Factor PSO", 
            "Velocity-Clamped PSO", 
            "PSO with Mutation",
            "Auto-Parameter Tuning"
        ])
        
        # Function to run K-Means
        def run_kmeans(X, k):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
            return labels, centroids, inertia, silhouette
          # Function to create standardized plots
        def create_cluster_plots(labels, centroids, X, X_scaled, features, scaler):
            # Create figures
            fig_2d, ax_2d = plt.subplots(figsize=(8, 6))
            
            # If we have 3 or more features, create interactive 3D plot
            if len(features) >= 3:
                # Create Plotly 3D scatter plot
                fig_3d = go.Figure()
                
                # Add data points colored by cluster
                colors = plt.cm.viridis(np.linspace(0, 1, k))
                
                # Add scatter points for each cluster
                for i in range(k):
                    cluster_points = X[labels == i]
                    if len(cluster_points) > 0:
                        fig_3d.add_trace(go.Scatter3d(
                            x=cluster_points[:, 0],
                            y=cluster_points[:, 1],
                            z=cluster_points[:, 2],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=f'rgba({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)},{colors[i][3]})',
                                opacity=0.7
                            ),
                            name=f'Cluster {i}'
                        ))
                
                # Add centroids
                fig_3d.add_trace(go.Scatter3d(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    z=centroids[:, 2],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='x'
                    ),
                    name='Centroids'
                ))
                
                # Update layout
                fig_3d.update_layout(
                    title="3D Cluster Visualization (Drag to Rotate)",
                    scene=dict(
                        xaxis_title=features[0],
                        yaxis_title=features[1],
                        zaxis_title=features[2]
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                    legend=dict(
                        x=0,
                        y=1
                    )
                )
            else:
                fig_3d = None
            
            # Plot 2D (always show first 2 features)
            ax_2d.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.7, cmap='viridis')
            ax_2d.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red')
            ax_2d.set_xlabel(features[0])
            ax_2d.set_ylabel(features[1])
            ax_2d.set_title("2D Cluster Visualization")
            
            # If we have PCA projection
            if len(features) > 2:
                pca = PCA(n_components=2, random_state=random_state)
                X_pca = pca.fit_transform(X_scaled)
                centroids_scaled = scaler.transform(centroids) if centroids.shape[1] == len(features) else centroids
                centroids_pca = pca.transform(centroids_scaled)
                
                fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
                ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.7, cmap='viridis')
                ax_pca.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red')
                ax_pca.set_xlabel("PC1")
                ax_pca.set_ylabel("PC2")
                ax_pca.set_title("PCA Projection with Clusters")
            else:
                fig_pca = None
            
            return fig_2d, fig_3d, fig_pca
          # 1. K-Means Tab
        with tab1:
            st.header("K-Means Clustering (Baseline)")
            st.markdown("""
            **Description:** 
            
            K-Means is a traditional clustering algorithm that partitions data into k clusters by minimizing the within-cluster variance.
            
            **How it works:**
            1. Initialize k centroids randomly
            2. Assign each data point to the nearest centroid
            3. Update centroids as the mean of assigned points
            4. Repeat steps 2-3 until convergence
            
            **Parameters:**
            - **Number of clusters (k)**: Determines how many clusters the data will be divided into. Higher values create more granular clusters, while lower values create broader clusters. Choosing the optimal k is often problem-dependent.
            - **Random seed**: Controls the initial placement of centroids. Different seeds can lead to different final clustering solutions, as K-Means is sensitive to initialization.
            
            **Pros:**
            - Simple and intuitive algorithm
            - Efficient for large datasets
            
            **Cons:**
            - Sensitive to initial centroid placement
            - May get stuck in local minima
            - Pre-defined number of clusters required
            """)
            
            # Button to run K-Means
            if st.button("Run K-Means Clustering", key="kmeans_run"):
                with st.spinner("Running K-Means..."):
                    # Run K-Means
                    labels, centroids, inertia, silhouette = run_kmeans(X_scaled, k)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Inertia (lower is better)", f"{inertia:.2f}")
                    with col2:
                        st.metric("Silhouette Score (higher is better)", f"{silhouette:.4f}")
                    
                    # Calculate original space centroids
                    centroids_orig = scaler.inverse_transform(centroids)
                    
                    # Create cluster visualization
                    fig_2d, fig_3d, fig_pca = create_cluster_plots(
                        labels, centroids_orig, X, X_scaled, features, scaler
                    )
                    
                    # Display plots
                    st.pyplot(fig_2d)
                    if fig_3d:
                        st.subheader("Interactive 3D Visualization (Drag to Rotate)")
                        st.plotly_chart(fig_3d, use_container_width=True)
                    if fig_pca:
                        st.pyplot(fig_pca)
                    
                    # Display cluster statistics
                    st.subheader("Cluster Statistics")
                    cluster_stats = pd.DataFrame({
                        'Feature': features * k,
                        'Cluster': np.repeat(range(k), len(features)),
                        'Centroid Value': centroids_orig.flatten()
                    })
                    st.dataframe(cluster_stats)
                    
                    # Save results for comparison
                    st.session_state.kmeans_results = {
                        'labels': labels,
                        'centroids': centroids,
                        'inertia': inertia,
                        'silhouette': silhouette
                    }
          # 2. Global-Best PSO Tab
        with tab2:
            st.header("Global-Best PSO Clustering")
            st.markdown("""
            **Description:**
            
            Global-Best PSO is the standard implementation of Particle Swarm Optimization for clustering. Each particle (potential solution) 
            is influenced by its own best position and the global best position found by any particle.
            
            **How it works:**
            1. Initialize particles randomly in the search space (each particle represents a set of centroids)
            2. For each iteration:
               - Update particle velocities based on personal best and global best positions
               - Update particle positions
               - Evaluate fitness (sum of squared distances)
               - Update personal and global best positions
            3. Final global best position represents the best centroids found
            
            **Parameters:**
            - **Swarm size**: The number of particles in the swarm. Larger swarms explore more of the search space but require more computation. Typically 20-50 particles provide good results.
            - **Max iterations**: Controls how long the algorithm runs. More iterations allow for better convergence but increase computation time. Should be set high enough to reach convergence.
            - **Inertia weight (w)**: Controls how much of the previous velocity is retained. Higher values (0.8-1.0) favor exploration, while lower values (0.2-0.5) favor exploitation. Typical range is 0.4-0.9.
            - **Cognitive coefficient (c1)**: Controls the influence of the particle's personal best. Higher values make particles more "nostalgic" and likely to return to their personal best positions. Typically set around 1.5-2.0.
            - **Social coefficient (c2)**: Controls the influence of the global best. Higher values make particles more attracted to the global best position. Typically set around 1.5-2.0.
            
            **Pros:**
            - Can escape local minima better than K-Means
            - No need for good initialization
            
            **Cons:**
            - May converge prematurely to local optima
            - More computationally expensive than K-Means
            """)
            
            # Parameters for Global-Best PSO
            col1, col2, col3 = st.columns(3)
            with col1:
                swarm_size_gb = st.slider("Swarm size", min_value=10, max_value=100, value=30, key="swarm_size_gb")
                w_gb = st.slider("Inertia weight (w)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="w_gb")
            with col2:
                max_iters_gb = st.slider("Max iterations", min_value=10, max_value=500, value=100, key="max_iters_gb")
                c1_gb = st.slider("Cognitive coefficient (c1)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c1_gb")
            with col3:
                c2_gb = st.slider("Social coefficient (c2)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c2_gb")
            
            # Button to run Global-Best PSO
            if st.button("Run Global-Best PSO Clustering", key="gb_pso_run"):
                with st.spinner("Running Global-Best PSO..."):
                    # Run Global-Best PSO
                    labels, centroids = pso_global(
                        X_scaled, k, swarm_size=swarm_size_gb, max_iters=max_iters_gb, 
                        w=w_gb, c1=c1_gb, c2=c2_gb
                    )
                    
                    # Calculate metrics
                    inertia = np.sum((X_scaled - centroids[labels])**2)
                    silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Inertia (lower is better)", f"{inertia:.2f}")
                    with col2:
                        st.metric("Silhouette Score (higher is better)", f"{silhouette:.4f}")
                    
                    # Calculate original space centroids
                    centroids_orig = scaler.inverse_transform(centroids)
                    
                    # Create cluster visualization
                    fig_2d, fig_3d, fig_pca = create_cluster_plots(
                        labels, centroids_orig, X, X_scaled, features, scaler
                    )
                    
                    # Display plots
                    st.pyplot(fig_2d)
                    if fig_3d:
                        st.subheader("Interactive 3D Visualization (Drag to Rotate)")
                        st.plotly_chart(fig_3d, use_container_width=True)
                    if fig_pca:
                        st.pyplot(fig_pca)
                    
                    # Display cluster statistics
                    st.subheader("Cluster Statistics")
                    cluster_stats = pd.DataFrame({
                        'Feature': features * k,
                        'Cluster': np.repeat(range(k), len(features)),
                        'Centroid Value': centroids_orig.flatten()
                    })
                    st.dataframe(cluster_stats)
                    
                    # Save results for comparison
                    st.session_state.gb_pso_results = {
                        'labels': labels,
                        'centroids': centroids,
                        'inertia': inertia,
                        'silhouette': silhouette
                    }
          # 3. Local-Best PSO Tab
        with tab3:
            st.header("Local-Best PSO Clustering")
            st.markdown("""
            **Description:**
            
            Local-Best PSO uses a neighborhood topology (ring topology) where each particle is influenced by its immediate neighbors 
            rather than the global best. This helps maintain diversity and can avoid premature convergence.
            
            **How it works:**
            1. Initialize particles randomly in the search space
            2. For each iteration:
               - Update particle velocities based on personal best and local best positions in neighborhood
               - Update particle positions
               - Evaluate fitness
               - Update personal and local best positions
            3. Final best position represents the best centroids found
            
            **Parameters:**
            - **Swarm size**: The number of particles in the swarm. Larger swarms provide more diversity in the population, which is especially beneficial for Local-Best PSO's neighborhood structure. Typically 30-50 particles work well.
            - **Max iterations**: Controls how long the algorithm runs. Local-Best PSO may need more iterations than Global-Best PSO to converge due to its slower information propagation through the swarm.
            - **Inertia weight (w)**: Controls momentum of particles. In Local-Best PSO, slightly higher values (0.6-0.8) often work better to maintain exploration since local neighborhoods already restrict information flow.
            - **Cognitive coefficient (c1)**: Controls the influence of the particle's personal best. In Local-Best PSO, balancing c1 and c2 is important for proper neighborhood influence.
            - **Social coefficient (c2)**: Controls the influence of the neighborhood best. Should be balanced with c1, but slightly higher values can help information propagate through neighborhoods.
            
            **Pros:**
            - Better exploration of the search space
            - Less likely to converge prematurely
            
            **Cons:**
            - Slower convergence than Global-Best PSO
            - More complex to implement
            """)
            
            # Parameters for Local-Best PSO
            col1, col2, col3 = st.columns(3)
            with col1:
                swarm_size_lb = st.slider("Swarm size", min_value=10, max_value=100, value=30, key="swarm_size_lb")
                w_lb = st.slider("Inertia weight (w)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="w_lb")
            with col2:
                max_iters_lb = st.slider("Max iterations", min_value=10, max_value=500, value=100, key="max_iters_lb")
                c1_lb = st.slider("Cognitive coefficient (c1)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c1_lb")
            with col3:
                c2_lb = st.slider("Social coefficient (c2)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c2_lb")
            
            # Button to run Local-Best PSO
            if st.button("Run Local-Best PSO Clustering", key="lb_pso_run"):
                with st.spinner("Running Local-Best PSO..."):
                    # Run Local-Best PSO
                    labels, centroids = pso_local(
                        X_scaled, k, swarm_size=swarm_size_lb, max_iters=max_iters_lb, 
                        w=w_lb, c1=c1_lb, c2=c2_lb
                    )
                    
                    # Calculate metrics
                    inertia = np.sum((X_scaled - centroids[labels])**2)
                    silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Inertia (lower is better)", f"{inertia:.2f}")
                    with col2:
                        st.metric("Silhouette Score (higher is better)", f"{silhouette:.4f}")
                    
                    # Calculate original space centroids
                    centroids_orig = scaler.inverse_transform(centroids)
                    
                    # Create cluster visualization
                    fig_2d, fig_3d, fig_pca = create_cluster_plots(
                        labels, centroids_orig, X, X_scaled, features, scaler
                    )
                    
                    # Display plots
                    st.pyplot(fig_2d)
                    if fig_3d:
                        st.subheader("Interactive 3D Visualization (Drag to Rotate)")
                        st.plotly_chart(fig_3d, use_container_width=True)
                    if fig_pca:
                        st.pyplot(fig_pca)
                    
                    # Display cluster statistics
                    st.subheader("Cluster Statistics")
                    cluster_stats = pd.DataFrame({
                        'Feature': features * k,
                        'Cluster': np.repeat(range(k), len(features)),
                        'Centroid Value': centroids_orig.flatten()
                    })
                    st.dataframe(cluster_stats)
                    
                    # Save results for comparison
                    st.session_state.lb_pso_results = {
                        'labels': labels,
                        'centroids': centroids,
                        'inertia': inertia,
                        'silhouette': silhouette
                    }
          # 4. Linear Inertia PSO Tab
        with tab4:
            st.header("Linear Inertia PSO Clustering")
            st.markdown("""
            **Description:**
            
            Linear Inertia PSO modifies the inertia weight parameter to linearly decrease from a high value to a low value over iterations. 
            This promotes exploration in early iterations and exploitation in later iterations.
            
            **How it works:**
            1. Initialize particles randomly
            2. For each iteration:
               - Update inertia weight (w) linearly from w_max to w_min
               - Update particle velocities with decreasing inertia
               - Update particle positions
               - Evaluate fitness
               - Update personal and global best positions
            3. Final global best position represents the best centroids found
            
            **Parameters:**
            - **Swarm size**: The number of particles in the swarm. Similar to standard PSO, but with Linear Inertia, smaller swarms (20-30) can sometimes work well since the adaptive inertia helps maintain diversity.
            - **Max iterations**: Controls how long the algorithm runs. This parameter is particularly important for Linear Inertia PSO as it determines the schedule for decreasing the inertia weight.
            - **Max inertia weight (w_max)**: The starting inertia weight value. Higher values (0.8-0.9) at the beginning promote exploration of the search space. This should be set significantly higher than w_min.
            - **Min inertia weight (w_min)**: The final inertia weight value. Lower values (0.2-0.4) encourage exploitation near the end of the search process.
            - **Cognitive coefficient (c1)**: Controls the influence of the particle's personal best. In Linear Inertia PSO, this often works well at standard values (1.5-2.0).
            - **Social coefficient (c2)**: Controls the influence of the global best. Should generally balance with c1, but slightly higher values can work well as inertia decreases.
            
            **Pros:**
            - Good balance between exploration and exploitation
            - Often produces better solutions than standard PSO
            
            **Cons:**
            - Requires tuning of w_max and w_min
            - Still may converge to local optima
            """)
            
            # Parameters for Linear Inertia PSO
            col1, col2, col3 = st.columns(3)
            with col1:
                swarm_size_li = st.slider("Swarm size", min_value=10, max_value=100, value=30, key="swarm_size_li")
                w_max_li = st.slider("Max inertia weight (w_max)", min_value=0.4, max_value=1.0, value=0.9, step=0.1, key="w_max_li")
            with col2:
                max_iters_li = st.slider("Max iterations", min_value=10, max_value=500, value=100, key="max_iters_li")
                w_min_li = st.slider("Min inertia weight (w_min)", min_value=0.1, max_value=0.6, value=0.4, step=0.1, key="w_min_li")
            with col3:
                c1_li = st.slider("Cognitive coefficient (c1)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c1_li")
                c2_li = st.slider("Social coefficient (c2)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c2_li")
            
            # Button to run Linear Inertia PSO
            if st.button("Run Linear Inertia PSO Clustering", key="li_pso_run"):
                with st.spinner("Running Linear Inertia PSO..."):
                    # Run Linear Inertia PSO
                    labels, centroids = pso_linear_inertia(
                        X_scaled, k, swarm_size=swarm_size_li, max_iters=max_iters_li, 
                        w_max=w_max_li, w_min=w_min_li, c1=c1_li, c2=c2_li
                    )
                    
                    # Calculate metrics
                    inertia = np.sum((X_scaled - centroids[labels])**2)
                    silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Inertia (lower is better)", f"{inertia:.2f}")
                    with col2:
                        st.metric("Silhouette Score (higher is better)", f"{silhouette:.4f}")
                    
                    # Calculate original space centroids
                    centroids_orig = scaler.inverse_transform(centroids)
                    
                    # Create cluster visualization
                    fig_2d, fig_3d, fig_pca = create_cluster_plots(
                        labels, centroids_orig, X, X_scaled, features, scaler
                    )
                    
                    # Display plots
                    st.pyplot(fig_2d)
                    if fig_3d:
                        st.subheader("Interactive 3D Visualization (Drag to Rotate)")
                        st.plotly_chart(fig_3d, use_container_width=True)
                    if fig_pca:
                        st.pyplot(fig_pca)
                    
                    # Display cluster statistics
                    st.subheader("Cluster Statistics")
                    cluster_stats = pd.DataFrame({
                        'Feature': features * k,
                        'Cluster': np.repeat(range(k), len(features)),
                        'Centroid Value': centroids_orig.flatten()
                    })
                    st.dataframe(cluster_stats)
                    
                    # Save results for comparison
                    st.session_state.li_pso_results = {
                        'labels': labels,
                        'centroids': centroids,
                        'inertia': inertia,
                        'silhouette': silhouette
                    }
          # 5. Constriction Factor PSO Tab
        with tab5:
            st.header("Constriction Factor PSO Clustering")
            st.markdown("""
            **Description:**
            
            Constriction Factor PSO uses a constriction coefficient derived from stability analysis to control particle velocity
            without requiring explicit velocity clamping. This provides theoretical convergence guarantees.
            
            **How it works:**
            1. Calculate constriction factor χ based on acceleration coefficients c1 and c2
            2. Initialize particles randomly
            3. For each iteration:
               - Update particle velocities using constriction factor
               - Update particle positions
               - Evaluate fitness
               - Update personal and global best positions
            4. Final global best position represents the best centroids found
            
            **Parameters:**
            - **Swarm size**: The number of particles in the swarm. For Constriction Factor PSO, smaller swarms (20-30) often work well since the constriction factor ensures convergence.
            - **Max iterations**: Controls how long the algorithm runs. Constriction Factor PSO typically requires fewer iterations than standard PSO due to its guaranteed convergence properties.
            - **Cognitive coefficient (c1)**: Controls the influence of the particle's personal best. In Constriction Factor PSO, both c1 and c2 are usually set higher (around 2.05 each) than in standard PSO.
            - **Social coefficient (c2)**: Controls the influence of the global best. Must be set such that φ = c1 + c2 > 4 for the constriction formula to work properly. Usually set around 2.05.
            
            **Pros:**
            - Theoretically guaranteed convergence
            - No need for velocity clamping or inertia tuning
            
            **Cons:**
            - Performance depends on proper setting of acceleration coefficients
            - May still get stuck in local optima
            """)
            
            # Parameters for Constriction Factor PSO
            col1, col2 = st.columns(2)
            with col1:
                swarm_size_cf = st.slider("Swarm size", min_value=10, max_value=100, value=30, key="swarm_size_cf")
                c1_cf = st.slider("Cognitive coefficient (c1)", min_value=1.5, max_value=2.5, value=2.05, step=0.05, key="c1_cf")
            with col2:
                max_iters_cf = st.slider("Max iterations", min_value=10, max_value=500, value=100, key="max_iters_cf")
                c2_cf = st.slider("Social coefficient (c2)", min_value=1.5, max_value=2.5, value=2.05, step=0.05, key="c2_cf")
            
            # Information about constriction factor
            st.info(f"The constriction factor χ = 2 / |2 - (c1 + c2) - √((c1 + c2)² - 4(c1 + c2))| = {2 / abs(2 - (c1_cf + c2_cf) - np.sqrt((c1_cf + c2_cf)**2 - 4*(c1_cf + c2_cf))):.4f}")
            
            # Button to run Constriction Factor PSO
            if st.button("Run Constriction Factor PSO Clustering", key="cf_pso_run"):
                with st.spinner("Running Constriction Factor PSO..."):
                    # Run Constriction Factor PSO
                    labels, centroids = pso_constriction(
                        X_scaled, k, swarm_size=swarm_size_cf, max_iters=max_iters_cf, 
                        c1=c1_cf, c2=c2_cf
                    )
                    
                    # Calculate metrics
                    inertia = np.sum((X_scaled - centroids[labels])**2)
                    silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Inertia (lower is better)", f"{inertia:.2f}")
                    with col2:
                        st.metric("Silhouette Score (higher is better)", f"{silhouette:.4f}")
                    
                    # Calculate original space centroids
                    centroids_orig = scaler.inverse_transform(centroids)
                    
                    # Create cluster visualization
                    fig_2d, fig_3d, fig_pca = create_cluster_plots(
                        labels, centroids_orig, X, X_scaled, features, scaler
                    )
                    
                    # Display plots
                    st.pyplot(fig_2d)
                    if fig_3d:
                        st.subheader("Interactive 3D Visualization (Drag to Rotate)")
                        st.plotly_chart(fig_3d, use_container_width=True)
                    if fig_pca:
                        st.pyplot(fig_pca)
                    
                    # Display cluster statistics
                    st.subheader("Cluster Statistics")
                    cluster_stats = pd.DataFrame({
                        'Feature': features * k,
                        'Cluster': np.repeat(range(k), len(features)),
                        'Centroid Value': centroids_orig.flatten()
                    })
                    st.dataframe(cluster_stats)
                    
                    # Save results for comparison
                    st.session_state.cf_pso_results = {
                        'labels': labels,
                        'centroids': centroids,
                        'inertia': inertia,
                        'silhouette': silhouette
                    }
        
        # 6. Velocity-Clamped PSO Tab
        with tab6:
            st.header("Velocity-Clamped PSO Clustering")
            st.markdown("""
            **Description:**
            
            Velocity-Clamped PSO limits the maximum velocity of particles to prevent them from taking excessive steps. 
            This helps control the search behavior and improves convergence.
            
            **How it works:**
            1. Initialize particles randomly
            2. Define velocity bounds (vmax) as a fraction of the data range
            3. For each iteration:
               - Update particle velocities
               - Clamp velocities to vmax
               - Update particle positions
               - Evaluate fitness
               - Update personal and global best positions
            4. Final global best position represents the best centroids found
            
            **Parameters:**
            - **Swarm size**: The number of particles in the swarm. Similar to standard PSO, but with velocity clamping a moderate swarm size (25-40) often works well.
            - **Max iterations**: Controls how long the algorithm runs. Velocity-Clamped PSO may require fewer iterations than standard PSO since particles are prevented from making excessively large steps.
            - **Inertia weight (w)**: Controls how much of the previous velocity is retained. With velocity clamping, moderate values (0.5-0.7) are often effective.
            - **Cognitive coefficient (c1)**: Controls the influence of the particle's personal best. Standard values around 1.5-2.0 work well with velocity clamping.
            - **Social coefficient (c2)**: Controls the influence of the global best. Should be balanced with c1, typically around 1.5-2.0.
            - **Velocity max fraction**: Defines the maximum velocity as a fraction of the data range. This is crucial for controlling particle movement - too low (0.1) restricts exploration, too high (0.5) allows more erratic movement. Typically values between 0.1-0.3 work well for clustering.
            
            **Pros:**
            - Prevents erratic particle movements
            - Better control over exploration vs. exploitation
            
            **Cons:**
            - Requires tuning of vmax parameter
            - Can restrict exploration if set too low
            """)
            
            # Parameters for Velocity-Clamped PSO
            col1, col2, col3 = st.columns(3)
            with col1:
                swarm_size_vc = st.slider("Swarm size", min_value=10, max_value=100, value=30, key="swarm_size_vc")
                w_vc = st.slider("Inertia weight (w)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="w_vc")
            with col2:
                max_iters_vc = st.slider("Max iterations", min_value=10, max_value=500, value=100, key="max_iters_vc")
                c1_vc = st.slider("Cognitive coefficient (c1)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c1_vc")
            with col3:
                c2_vc = st.slider("Social coefficient (c2)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c2_vc")
                vmax_frac = st.slider("Velocity max fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05, key="vmax_frac")
            
            # Button to run Velocity-Clamped PSO
            if st.button("Run Velocity-Clamped PSO Clustering", key="vc_pso_run"):
                with st.spinner("Running Velocity-Clamped PSO..."):
                    # Run Velocity-Clamped PSO
                    labels, centroids = pso_velocity_clamped(
                        X_scaled, k, swarm_size=swarm_size_vc, max_iters=max_iters_vc, 
                        w=w_vc, c1=c1_vc, c2=c2_vc, vmax_frac=vmax_frac
                    )
                    
                    # Calculate metrics
                    inertia = np.sum((X_scaled - centroids[labels])**2)
                    silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Inertia (lower is better)", f"{inertia:.2f}")
                    with col2:
                        st.metric("Silhouette Score (higher is better)", f"{silhouette:.4f}")
                    
                    # Calculate original space centroids
                    centroids_orig = scaler.inverse_transform(centroids)
                    
                    # Create cluster visualization
                    fig_2d, fig_3d, fig_pca = create_cluster_plots(
                        labels, centroids_orig, X, X_scaled, features, scaler
                    )
                    
                    # Display plots
                    st.pyplot(fig_2d)
                    if fig_3d:
                        st.subheader("Interactive 3D Visualization (Drag to Rotate)")
                        st.plotly_chart(fig_3d, use_container_width=True)
                    if fig_pca:
                        st.pyplot(fig_pca)
                    
                    # Display cluster statistics
                    st.subheader("Cluster Statistics")
                    cluster_stats = pd.DataFrame({
                        'Feature': features * k,
                        'Cluster': np.repeat(range(k), len(features)),
                        'Centroid Value': centroids_orig.flatten()
                    })
                    st.dataframe(cluster_stats)
                    
                    # Save results for comparison
                    st.session_state.vc_pso_results = {
                        'labels': labels,
                        'centroids': centroids,
                        'inertia': inertia,
                        'silhouette': silhouette
                    }
        
        # 7. PSO with Mutation Tab
        with tab7:
            st.header("PSO with Gaussian Mutation")
            st.markdown("""            **Description:**
            
            PSO with Gaussian Mutation introduces random perturbations to some particles during the search. 
            This additional randomness helps avoid premature convergence and enhances exploration.
            
            **How it works:**
            1. Initialize particles randomly
            2. For each iteration:
               - Update particle velocities and positions as in standard PSO
               - With a certain probability, add Gaussian noise to particle positions
               - Evaluate fitness
               - Update personal and global best positions
            3. Final global best position represents the best centroids found
            
            **Parameters:**
            - **Swarm size**: The number of particles in the swarm. For PSO with Gaussian Mutation, a moderate swarm size (25-40) often works well. With mutation adding diversity, smaller swarms can sometimes perform effectively.
            - **Max iterations**: Controls how long the algorithm runs. PSO with Gaussian Mutation may require more iterations to fully converge due to the added randomness.
            - **Inertia weight (w)**: Controls how much of the previous velocity is retained. Standard values around 0.5-0.7 work well with mutation, as the mutation adds additional exploration capability.
            - **Cognitive coefficient (c1)**: Controls the influence of the particle's personal best. Standard values around 1.5-2.0 work well.
            - **Social coefficient (c2)**: Controls the influence of the global best. Standard values around 1.5-2.0 work well.
            - **Mutation probability**: Determines how often particles undergo mutation. Higher values (0.3-0.5) increase exploration but may slow convergence, while lower values (0.05-0.15) provide occasional helpful randomness. Critical for escaping local optima.
            - **Mutation scale**: Controls the magnitude of the Gaussian noise added during mutation. Larger values cause more significant position changes. Typically small values (0.01-0.1) work best, with smaller values preferred as the algorithm progresses.
            
            **Pros:**
            - Enhanced exploration capability
            - Better chance of escaping local optima
            
            **Cons:**
            - Requires tuning of mutation probability and scale
            - May slow convergence for simple problems
            """)
            
            # Parameters for PSO with Mutation
            col1, col2, col3 = st.columns(3)
            with col1:
                swarm_size_m = st.slider("Swarm size", min_value=10, max_value=100, value=30, key="swarm_size_m")
                w_m = st.slider("Inertia weight (w)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="w_m")
                mutation_prob = st.slider("Mutation probability", min_value=0.05, max_value=0.5, value=0.2, step=0.05, key="mutation_prob")
            with col2:
                max_iters_m = st.slider("Max iterations", min_value=10, max_value=500, value=100, key="max_iters_m")
                c1_m = st.slider("Cognitive coefficient (c1)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c1_m")
                mutation_scale = st.slider("Mutation scale", min_value=0.01, max_value=0.2, value=0.05, step=0.01, key="mutation_scale")
            with col3:
                c2_m = st.slider("Social coefficient (c2)", min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="c2_m")
            
            # Button to run PSO with Mutation
            if st.button("Run PSO with Mutation", key="m_pso_run"):
                with st.spinner("Running PSO with Mutation..."):
                    # Run PSO with Mutation
                    labels, centroids = pso_mutation(
                        X_scaled, k, swarm_size=swarm_size_m, max_iters=max_iters_m, 
                        w=w_m, c1=c1_m, c2=c2_m, mutation_prob=mutation_prob, mutation_scale=mutation_scale
                    )
                    
                    # Calculate metrics
                    inertia = np.sum((X_scaled - centroids[labels])**2)
                    silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Inertia (lower is better)", f"{inertia:.2f}")
                    with col2:
                        st.metric("Silhouette Score (higher is better)", f"{silhouette:.4f}")
                    
                    # Calculate original space centroids
                    centroids_orig = scaler.inverse_transform(centroids)
                    
                    # Create cluster visualization
                    fig_2d, fig_3d, fig_pca = create_cluster_plots(
                        labels, centroids_orig, X, X_scaled, features, scaler
                    )
                    
                    # Display plots
                    st.pyplot(fig_2d)
                    if fig_3d:
                        st.subheader("Interactive 3D Visualization (Drag to Rotate)")
                        st.plotly_chart(fig_3d, use_container_width=True)
                    if fig_pca:
                        st.pyplot(fig_pca)
                    
                    # Display cluster statistics
                    st.subheader("Cluster Statistics")
                    cluster_stats = pd.DataFrame({
                        'Feature': features * k,
                        'Cluster': np.repeat(range(k), len(features)),
                        'Centroid Value': centroids_orig.flatten()
                    })
                    st.dataframe(cluster_stats)
                      # Save results for comparison
                    st.session_state.m_pso_results = {
                        'labels': labels,
                        'centroids': centroids,
                        'inertia': inertia,
                        'silhouette': silhouette
                    }
        
        # 8. Auto-Parameter Tuning Tab
        with tab8:
            st.header("Auto-Parameter Tuning")
            st.markdown("""            **Description:**
            
            This tab automatically tunes parameters for all PSO variations and K-Means to find the best configuration for each algorithm.
            The system will run multiple iterations with different parameter combinations and select the best performing set.
            
            **How it works:**
            1. Define parameter ranges for each algorithm
            2. Run each algorithm with different parameter combinations
            3. Evaluate results using silhouette score and inertia
            4. Select the best parameter set for each algorithm
            5. Compare all algorithms with their best parameters
            
            **Parameters:**
            - **Number of runs per parameter set**: Controls how many times each parameter combination is tested. Higher values (5-10) provide more reliable results by averaging out randomness, but significantly increase computation time. Lower values (1-3) are faster but may be influenced by lucky/unlucky initializations.
            - **Optimization metric**: Determines which performance measure is used to select the best parameters. "Silhouette Score" prioritizes well-separated, compact clusters, while "Inertia" focuses on minimizing within-cluster distances. Choose based on your clustering objectives.
            - **Number of parameter variations to try**: Controls how many different values to test for each parameter. Higher values (4-5) explore the parameter space more thoroughly but increase computation time exponentially. Lower values (2-3) provide faster results but may miss optimal settings.
            
            **Benefits:**
            - Automatically finds optimal parameters without manual tuning
            - Provides fair comparison between algorithms at their best performance
            - Shows stability and robustness across multiple runs
            """)
            
            # Parameters for auto-tuning
            col1, col2 = st.columns(2)
            with col1:
                num_runs = st.slider("Number of runs per parameter set", min_value=1, max_value=10, value=3, key="num_runs")
                auto_metric = st.selectbox("Optimization metric", ["Silhouette Score", "Inertia"], key="auto_metric")
            with col2:
                param_variations = st.slider("Number of parameter variations to try", min_value=2, max_value=5, value=3, key="param_variations")
            
            # Define parameter grids for each algorithm
            def get_parameter_grids(variations):
                # K-Means doesn't have many parameters to tune
                kmeans_grid = {
                    'random_state': list(range(10, 10+variations))
                }
                
                # Global-Best PSO
                gb_pso_grid = {
                    'swarm_size': [20 + 10*i for i in range(variations)],
                    'max_iters': [50 + 50*i for i in range(variations)],
                    'w': [0.3 + 0.2*i for i in range(variations)],
                    'c1': [1.0 + 0.5*i for i in range(variations)],
                    'c2': [1.0 + 0.5*i for i in range(variations)]
                }
                
                # Local-Best PSO
                lb_pso_grid = {
                    'swarm_size': [20 + 10*i for i in range(variations)],
                    'max_iters': [50 + 50*i for i in range(variations)],
                    'w': [0.3 + 0.2*i for i in range(variations)],
                    'c1': [1.0 + 0.5*i for i in range(variations)],
                    'c2': [1.0 + 0.5*i for i in range(variations)]
                }
                
                # Linear Inertia PSO
                li_pso_grid = {
                    'swarm_size': [20 + 10*i for i in range(variations)],
                    'max_iters': [50 + 50*i for i in range(variations)],
                    'w_max': [0.7 + 0.1*i for i in range(variations)],
                    'w_min': [0.2 + 0.1*i for i in range(variations)],
                    'c1': [1.0 + 0.5*i for i in range(variations)],
                    'c2': [1.0 + 0.5*i for i in range(variations)]
                }
                
                # Constriction Factor PSO
                cf_pso_grid = {
                    'swarm_size': [20 + 10*i for i in range(variations)],
                    'max_iters': [50 + 50*i for i in range(variations)],
                    'c1': [2.0 + 0.05*i for i in range(variations)],
                    'c2': [2.0 + 0.05*i for i in range(variations)]
                }
                
                # Velocity-Clamped PSO
                vc_pso_grid = {
                    'swarm_size': [20 + 10*i for i in range(variations)],
                    'max_iters': [50 + 50*i for i in range(variations)],
                    'w': [0.3 + 0.2*i for i in range(variations)],
                    'c1': [1.0 + 0.5*i for i in range(variations)],
                    'c2': [1.0 + 0.5*i for i in range(variations)],
                    'vmax_frac': [0.1 + 0.1*i for i in range(variations)]
                }
                
                # PSO with Mutation
                m_pso_grid = {
                    'swarm_size': [20 + 10*i for i in range(variations)],
                    'max_iters': [50 + 50*i for i in range(variations)],
                    'w': [0.3 + 0.2*i for i in range(variations)],
                    'c1': [1.0 + 0.5*i for i in range(variations)],
                    'c2': [1.0 + 0.5*i for i in range(variations)],
                    'mutation_prob': [0.1 + 0.1*i for i in range(variations)],
                    'mutation_scale': [0.03 + 0.02*i for i in range(variations)]
                }
                
                return {
                    'K-Means': kmeans_grid,
                    'Global-Best PSO': gb_pso_grid,
                    'Local-Best PSO': lb_pso_grid,
                    'Linear Inertia PSO': li_pso_grid,
                    'Constriction Factor PSO': cf_pso_grid,
                    'Velocity-Clamped PSO': vc_pso_grid,
                    'PSO with Mutation': m_pso_grid
                }
            
            # Function to run parameter tuning
            def run_parameter_tuning(X_scaled, k, num_runs, param_variations, metric='Silhouette Score'):
                results = []
                parameter_grids = get_parameter_grids(param_variations)
                
                progress_bar = st.progress(0)
                total_algorithms = 7
                algorithm_count = 0
                
                # Helper function to update progress
                def update_progress(alg_progress):
                    nonlocal algorithm_count
                    progress = (algorithm_count + alg_progress) / total_algorithms
                    progress_bar.progress(progress)
                
                # 1. K-Means Tuning
                st.write("Tuning K-Means...")
                best_kmeans = {'score': -np.inf if metric == 'Silhouette Score' else np.inf}
                best_comparison = None
                
                for random_state in parameter_grids['K-Means']['random_state']:
                    for _ in range(num_runs):
                        kmeans = KMeans(n_clusters=k, random_state=random_state)
                        labels = kmeans.fit_predict(X_scaled)
                        inertia = kmeans.inertia_
                        silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                        
                        score = silhouette if metric == 'Silhouette Score' else -inertia
                        if (metric == 'Silhouette Score' and score > best_kmeans['score']) or \
                           (metric == 'Inertia' and -score < best_kmeans['score']):
                            best_kmeans = {
                                'score': score,
                                'params': {'random_state': random_state},
                                'labels': labels,
                                'centroids': kmeans.cluster_centers_,
                                'inertia': inertia,
                                'silhouette': silhouette
                            }
                
                results.append({
                    'Algorithm': 'K-Means',
                    'Best Params': best_kmeans['params'],
                    'Inertia': best_kmeans['inertia'],
                    'Silhouette Score': best_kmeans['silhouette'],
                    'Labels': best_kmeans['labels'],
                    'Centroids': best_kmeans['centroids']
                })
                
                update_progress(1/total_algorithms)
                algorithm_count += 1
                
                # 2. Global-Best PSO Tuning
                st.write("Tuning Global-Best PSO...")
                best_gb_pso = {'score': -np.inf if metric == 'Silhouette Score' else np.inf}
                
                # Generate parameter combinations
                from itertools import product
                gb_keys = parameter_grids['Global-Best PSO'].keys()
                gb_values = parameter_grids['Global-Best PSO'].values()
                
                for items in product(*gb_values):
                    params = dict(zip(gb_keys, items))
                    run_progress = 0
                    
                    for run in range(num_runs):
                        labels, centroids = pso_global(
                            X_scaled, k, 
                            swarm_size=params['swarm_size'], 
                            max_iters=params['max_iters'],
                            w=params['w'], 
                            c1=params['c1'], 
                            c2=params['c2']
                        )
                        
                        inertia = np.sum((X_scaled - centroids[labels])**2)
                        silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                        
                        score = silhouette if metric == 'Silhouette Score' else -inertia
                        if (metric == 'Silhouette Score' and score > best_gb_pso['score']) or \
                           (metric == 'Inertia' and -score < best_gb_pso['score']):
                            best_gb_pso = {
                                'score': score,
                                'params': params,
                                'labels': labels,
                                'centroids': centroids,
                                'inertia': inertia,
                                'silhouette': silhouette
                            }
                        
                        run_progress = (run + 1) / num_runs
                        update_progress(run_progress / len(list(product(*gb_values))))
                
                results.append({
                    'Algorithm': 'Global-Best PSO',
                    'Best Params': best_gb_pso['params'],
                    'Inertia': best_gb_pso['inertia'],
                    'Silhouette Score': best_gb_pso['silhouette'],
                    'Labels': best_gb_pso['labels'],
                    'Centroids': best_gb_pso['centroids']
                })
                
                update_progress(1/total_algorithms)
                algorithm_count += 1
                
                # 3. Local-Best PSO Tuning
                st.write("Tuning Local-Best PSO...")
                best_lb_pso = {'score': -np.inf if metric == 'Silhouette Score' else np.inf}
                
                # Generate parameter combinations
                lb_keys = parameter_grids['Local-Best PSO'].keys()
                lb_values = parameter_grids['Local-Best PSO'].values()
                
                for items in product(*lb_values):
                    params = dict(zip(lb_keys, items))
                    run_progress = 0
                    
                    for run in range(num_runs):
                        labels, centroids = pso_local(
                            X_scaled, k, 
                            swarm_size=params['swarm_size'], 
                            max_iters=params['max_iters'],
                            w=params['w'], 
                            c1=params['c1'], 
                            c2=params['c2']
                        )
                        
                        inertia = np.sum((X_scaled - centroids[labels])**2)
                        silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                        
                        score = silhouette if metric == 'Silhouette Score' else -inertia
                        if (metric == 'Silhouette Score' and score > best_lb_pso['score']) or \
                           (metric == 'Inertia' and -score < best_lb_pso['score']):
                            best_lb_pso = {
                                'score': score,
                                'params': params,
                                'labels': labels,
                                'centroids': centroids,
                                'inertia': inertia,
                                'silhouette': silhouette
                            }
                        
                        run_progress = (run + 1) / num_runs
                        update_progress(run_progress / len(list(product(*lb_values))))
                
                results.append({
                    'Algorithm': 'Local-Best PSO',
                    'Best Params': best_lb_pso['params'],
                    'Inertia': best_lb_pso['inertia'],
                    'Silhouette Score': best_lb_pso['silhouette'],
                    'Labels': best_lb_pso['labels'],
                    'Centroids': best_lb_pso['centroids']
                })
                
                update_progress(1/total_algorithms)
                algorithm_count += 1
                
                # 4. Linear Inertia PSO Tuning
                st.write("Tuning Linear Inertia PSO...")
                best_li_pso = {'score': -np.inf if metric == 'Silhouette Score' else np.inf}
                
                # Generate parameter combinations
                li_keys = parameter_grids['Linear Inertia PSO'].keys()
                li_values = parameter_grids['Linear Inertia PSO'].values()
                
                for items in product(*li_values):
                    params = dict(zip(li_keys, items))
                    
                    # Ensure w_max > w_min
                    if params['w_max'] <= params['w_min']:
                        continue
                    
                    run_progress = 0
                    
                    for run in range(num_runs):
                        labels, centroids = pso_linear_inertia(
                            X_scaled, k, 
                            swarm_size=params['swarm_size'], 
                            max_iters=params['max_iters'],
                            w_max=params['w_max'], 
                            w_min=params['w_min'], 
                            c1=params['c1'], 
                            c2=params['c2']
                        )
                        
                        inertia = np.sum((X_scaled - centroids[labels])**2)
                        silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                        
                        score = silhouette if metric == 'Silhouette Score' else -inertia
                        if (metric == 'Silhouette Score' and score > best_li_pso['score']) or \
                           (metric == 'Inertia' and -score < best_li_pso['score']):
                            best_li_pso = {
                                'score': score,
                                'params': params,
                                'labels': labels,
                                'centroids': centroids,
                                'inertia': inertia,
                                'silhouette': silhouette
                            }
                        
                        run_progress = (run + 1) / num_runs
                        update_progress(run_progress / len(list(product(*li_values))))
                
                results.append({
                    'Algorithm': 'Linear Inertia PSO',
                    'Best Params': best_li_pso['params'],
                    'Inertia': best_li_pso['inertia'],
                    'Silhouette Score': best_li_pso['silhouette'],
                    'Labels': best_li_pso['labels'],
                    'Centroids': best_li_pso['centroids']
                })
                
                update_progress(1/total_algorithms)
                algorithm_count += 1
                
                # 5. Constriction Factor PSO Tuning
                st.write("Tuning Constriction Factor PSO...")
                best_cf_pso = {'score': -np.inf if metric == 'Silhouette Score' else np.inf}
                
                # Generate parameter combinations
                cf_keys = parameter_grids['Constriction Factor PSO'].keys()
                cf_values = parameter_grids['Constriction Factor PSO'].values()
                
                for items in product(*cf_values):
                    params = dict(zip(cf_keys, items))
                    run_progress = 0
                    
                    for run in range(num_runs):
                        # Ensure phi > 4 for constriction factor formula
                        if params['c1'] + params['c2'] <= 4:
                            continue
                            
                        labels, centroids = pso_constriction(
                            X_scaled, k, 
                            swarm_size=params['swarm_size'], 
                            max_iters=params['max_iters'],
                            c1=params['c1'], 
                            c2=params['c2']
                        )
                        
                        inertia = np.sum((X_scaled - centroids[labels])**2)
                        silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                        
                        score = silhouette if metric == 'Silhouette Score' else -inertia
                        if (metric == 'Silhouette Score' and score > best_cf_pso['score']) or \
                           (metric == 'Inertia' and -score < best_cf_pso['score']):
                            best_cf_pso = {
                                'score': score,
                                'params': params,
                                'labels': labels,
                                'centroids': centroids,
                                'inertia': inertia,
                                'silhouette': silhouette
                            }
                        
                        run_progress = (run + 1) / num_runs
                        update_progress(run_progress / len(list(product(*cf_values))))
                
                results.append({
                    'Algorithm': 'Constriction Factor PSO',
                    'Best Params': best_cf_pso['params'],
                    'Inertia': best_cf_pso['inertia'],
                    'Silhouette Score': best_cf_pso['silhouette'],
                    'Labels': best_cf_pso['labels'],
                    'Centroids': best_cf_pso['centroids']
                })
                
                update_progress(1/total_algorithms)
                algorithm_count += 1
                
                # 6. Velocity-Clamped PSO Tuning
                st.write("Tuning Velocity-Clamped PSO...")
                best_vc_pso = {'score': -np.inf if metric == 'Silhouette Score' else np.inf}
                
                # Generate parameter combinations
                vc_keys = parameter_grids['Velocity-Clamped PSO'].keys()
                vc_values = parameter_grids['Velocity-Clamped PSO'].values()
                
                for items in product(*vc_values):
                    params = dict(zip(vc_keys, items))
                    run_progress = 0
                    
                    for run in range(num_runs):
                        labels, centroids = pso_velocity_clamped(
                            X_scaled, k, 
                            swarm_size=params['swarm_size'], 
                            max_iters=params['max_iters'],
                            w=params['w'], 
                            c1=params['c1'], 
                            c2=params['c2'],
                            vmax_frac=params['vmax_frac']
                        )
                        
                        inertia = np.sum((X_scaled - centroids[labels])**2)
                        silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                        
                        score = silhouette if metric == 'Silhouette Score' else -inertia
                        if (metric == 'Silhouette Score' and score > best_vc_pso['score']) or \
                           (metric == 'Inertia' and -score < best_vc_pso['score']):
                            best_vc_pso = {
                                'score': score,
                                'params': params,
                                'labels': labels,
                                'centroids': centroids,
                                'inertia': inertia,
                                'silhouette': silhouette
                            }
                        
                        run_progress = (run + 1) / num_runs
                        update_progress(run_progress / len(list(product(*vc_values))))
                
                results.append({
                    'Algorithm': 'Velocity-Clamped PSO',
                    'Best Params': best_vc_pso['params'],
                    'Inertia': best_vc_pso['inertia'],
                    'Silhouette Score': best_vc_pso['silhouette'],
                    'Labels': best_vc_pso['labels'],
                    'Centroids': best_vc_pso['centroids']
                })
                
                update_progress(1/total_algorithms)
                algorithm_count += 1
                
                # 7. PSO with Mutation Tuning
                st.write("Tuning PSO with Mutation...")
                best_m_pso = {'score': -np.inf if metric == 'Silhouette Score' else np.inf}
                
                # Generate parameter combinations
                m_keys = parameter_grids['PSO with Mutation'].keys()
                m_values = parameter_grids['PSO with Mutation'].values()
                
                for items in product(*m_values):
                    params = dict(zip(m_keys, items))
                    run_progress = 0
                    
                    for run in range(num_runs):
                        labels, centroids = pso_mutation(
                            X_scaled, k, 
                            swarm_size=params['swarm_size'], 
                            max_iters=params['max_iters'],
                            w=params['w'], 
                            c1=params['c1'], 
                            c2=params['c2'],
                            mutation_prob=params['mutation_prob'],
                            mutation_scale=params['mutation_scale']
                        )
                        
                        inertia = np.sum((X_scaled - centroids[labels])**2)
                        silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
                        
                        score = silhouette if metric == 'Silhouette Score' else -inertia
                        if (metric == 'Silhouette Score' and score > best_m_pso['score']) or \
                           (metric == 'Inertia' and -score < best_m_pso['score']):
                            best_m_pso = {
                                'score': score,
                                'params': params,
                                'labels': labels,
                                'centroids': centroids,
                                'inertia': inertia,
                                'silhouette': silhouette
                            }
                        
                        run_progress = (run + 1) / num_runs
                        update_progress(run_progress / len(list(product(*m_values))))
                
                results.append({
                    'Algorithm': 'PSO with Mutation',
                    'Best Params': best_m_pso['params'],
                    'Inertia': best_m_pso['inertia'],
                    'Silhouette Score': best_m_pso['silhouette'],
                    'Labels': best_m_pso['labels'],
                    'Centroids': best_m_pso['centroids']
                })
                
                update_progress(1.0)
                
                # Save all results in session state
                st.session_state.auto_tuning_results = results
                
                return results
            
            # Button to run auto-tuning
            if st.button("Run Auto-Parameter Tuning", key="auto_tune_run"):
                with st.spinner("Running auto-parameter tuning (this may take a while)..."):
                    # Run parameter tuning
                    results = run_parameter_tuning(
                        X_scaled, 
                        k, 
                        num_runs,
                        param_variations,
                        auto_metric
                    )
                    
                    # Display results
                    st.subheader("Parameter Tuning Results")
                    
                    # Create dataframe for metrics only (excluding params, labels, centroids)
                    df_metrics = pd.DataFrame([{
                        'Algorithm': r['Algorithm'],
                        'Inertia': r['Inertia'],
                        'Silhouette Score': r['Silhouette Score']
                    } for r in results])
                    
                    # Display metrics table
                    st.write("Performance Metrics:")
                    st.dataframe(df_metrics.round(4))
                    
                    # Find best algorithm
                    if auto_metric == 'Silhouette Score':
                        best_idx = df_metrics['Silhouette Score'].idxmax()
                    else:
                        best_idx = df_metrics['Inertia'].idxmin()
                    
                    best_algorithm = df_metrics.iloc[best_idx]['Algorithm']
                    
                    # Display best parameters for each algorithm
                    st.write("Best Parameters:")
                    for r in results:
                        with st.expander(f"{r['Algorithm']} Parameters"):
                            st.json(r['Best Params'])
                    
                    # Show visualization of best algorithm
                    st.subheader(f"Best Algorithm: {best_algorithm}")
                    best_result = results[best_idx]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Inertia", f"{best_result['Inertia']:.2f}")
                    with col2:
                        st.metric("Silhouette Score", f"{best_result['Silhouette Score']:.4f}")
                    
                    # Transform centroids back to original space
                    centroids_orig = scaler.inverse_transform(best_result['Centroids'])
                    
                    # Visualize best result
                    fig_2d, fig_3d, fig_pca = create_cluster_plots(
                        best_result['Labels'],
                        centroids_orig,
                        X,
                        X_scaled,
                        features,
                        scaler
                    )
                    
                    # Display plots
                    st.pyplot(fig_2d)
                    if fig_3d:
                        st.subheader("Interactive 3D Visualization (Drag to Rotate)")
                        st.plotly_chart(fig_3d, use_container_width=True)
                    if fig_pca:
                        st.pyplot(fig_pca)
                    
                    # Comparison of all algorithms
                    st.subheader("Algorithm Comparison")
                    
                    # Bar charts for comparing algorithms
                    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Inertia comparison (lower is better)
                    ax[0].barh(df_metrics['Algorithm'], df_metrics['Inertia'], color='skyblue')
                    ax[0].set_title('Inertia Comparison (Lower is Better)')
                    ax[0].set_xlabel('Inertia')
                    ax[0].invert_xaxis()  # So better values are further right
                    
                    # Silhouette comparison (higher is better)
                    ax[1].barh(df_metrics['Algorithm'], df_metrics['Silhouette Score'], color='coral')
                    ax[1].set_title('Silhouette Score Comparison (Higher is Better)')
                    ax[1].set_xlabel('Silhouette Score')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Save the tuned algorithm results to session state for comparison
                    for r in results:
                        algo_key = r['Algorithm'].lower().replace(' ', '_').replace('-', '_')
                        st.session_state[f"{algo_key}_tuned_results"] = {
                            'labels': r['Labels'],
                            'centroids': r['Centroids'],
                            'inertia': r['Inertia'],
                            'silhouette': r['Silhouette Score'],
                            'params': r['Best Params']
                        }
          # Comparison Tab (shown after running at least one algorithm)
        if any(f"{algo}_results" in st.session_state for algo in ['kmeans', 'gb_pso', 'lb_pso', 'li_pso', 'cf_pso', 'vc_pso', 'm_pso']) or \
           'auto_tuning_results' in st.session_state:
            st.header("Algorithm Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            
            # Add data for each algorithm if available
            if 'kmeans_results' in st.session_state:
                comparison_data.append({
                    'Algorithm': 'K-Means',
                    'Type': 'Manual',
                    'Inertia': st.session_state.kmeans_results['inertia'],
                    'Silhouette Score': st.session_state.kmeans_results['silhouette']
                })
            
            if 'gb_pso_results' in st.session_state:
                comparison_data.append({
                    'Algorithm': 'Global-Best PSO',
                    'Type': 'Manual',
                    'Inertia': st.session_state.gb_pso_results['inertia'],
                    'Silhouette Score': st.session_state.gb_pso_results['silhouette']
                })
            
            if 'lb_pso_results' in st.session_state:
                comparison_data.append({
                    'Algorithm': 'Local-Best PSO',
                    'Type': 'Manual',
                    'Inertia': st.session_state.lb_pso_results['inertia'],
                    'Silhouette Score': st.session_state.lb_pso_results['silhouette']
                })
            
            if 'li_pso_results' in st.session_state:
                comparison_data.append({
                    'Algorithm': 'Linear Inertia PSO',
                    'Type': 'Manual',
                    'Inertia': st.session_state.li_pso_results['inertia'],
                    'Silhouette Score': st.session_state.li_pso_results['silhouette']
                })
            
            if 'cf_pso_results' in st.session_state:
                comparison_data.append({
                    'Algorithm': 'Constriction Factor PSO',
                    'Type': 'Manual',
                    'Inertia': st.session_state.cf_pso_results['inertia'],
                    'Silhouette Score': st.session_state.cf_pso_results['silhouette']
                })
            
            if 'vc_pso_results' in st.session_state:
                comparison_data.append({
                    'Algorithm': 'Velocity-Clamped PSO',
                    'Type': 'Manual',
                    'Inertia': st.session_state.vc_pso_results['inertia'],
                    'Silhouette Score': st.session_state.vc_pso_results['silhouette']
                })
            
            if 'm_pso_results' in st.session_state:
                comparison_data.append({
                    'Algorithm': 'PSO with Mutation',
                    'Type': 'Manual',
                    'Inertia': st.session_state.m_pso_results['inertia'],
                    'Silhouette Score': st.session_state.m_pso_results['silhouette']
                })
            
            # Add auto-tuned results if available
            if 'auto_tuning_results' in st.session_state:
                for r in st.session_state.auto_tuning_results:
                    comparison_data.append({
                        'Algorithm': f"{r['Algorithm']} (Auto-Tuned)",
                        'Type': 'Auto-Tuned',
                        'Inertia': r['Inertia'],
                        'Silhouette Score': r['Silhouette Score']
                    })
            
            if comparison_data:
                # Create dataframe
                df_comparison = pd.DataFrame(comparison_data)
                
                # Display table
                st.subheader("Metrics Comparison")
                st.dataframe(df_comparison.round(4))
                
                # Bar charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    # Use different colors for manual vs auto-tuned
                    colors = ['skyblue' if row['Type'] == 'Manual' else 'navy' for _, row in df_comparison.iterrows()]
                    ax.barh(df_comparison['Algorithm'], df_comparison['Inertia'], color=colors)
                    ax.set_title('Inertia Comparison (Lower is Better)')
                    ax.set_xlabel('Inertia')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    # Use different colors for manual vs auto-tuned
                    colors = ['coral' if row['Type'] == 'Manual' else 'darkred' for _, row in df_comparison.iterrows()]
                    ax.barh(df_comparison['Algorithm'], df_comparison['Silhouette Score'], color=colors)
                    ax.set_title('Silhouette Score Comparison (Higher is Better)')
                    ax.set_xlabel('Silhouette Score')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Summary of findings
                st.subheader("Summary of Findings")
                
                # Find best algorithms
                best_inertia_algo = df_comparison.loc[df_comparison['Inertia'].idxmin(), 'Algorithm']
                best_silhouette_algo = df_comparison.loc[df_comparison['Silhouette Score'].idxmax(), 'Algorithm']
                
                st.markdown(f"""
                - **Best algorithm by Inertia:** {best_inertia_algo}
                - **Best algorithm by Silhouette Score:** {best_silhouette_algo}
                
                **Observations:**
                - PSO variations generally provide more robust solutions than standard K-Means
                - The best PSO variant for this dataset appears to be {best_silhouette_algo} based on silhouette score
                - Auto-tuned parameters often yield better results than manual parameter settings
                
                **Recommendations:**
                - For well-separated clusters: Standard Global-Best PSO may be sufficient
                - For complex or overlapping clusters: Consider Linear Inertia PSO or PSO with Mutation
                - For real-time applications: K-Means or Constriction Factor PSO may be preferable for speed
                - Always perform parameter tuning to get the best results for your specific dataset
                """)
    else:
        st.warning("Please provide a dataset to start clustering.")
else:
    st.warning("Please provide a dataset to start clustering.")

# Add footer
st.markdown("---")
st.markdown("PSO Clustering Variations Explorer | Built with Streamlit")
