# Comparative Analysis of Particle Swarm Optimization Variants for Clustering

## PSO Clustering Variations Explorer

An interactive study of Particle Swarm Optimization (PSO) variants for unsupervised clustering.

This project was developed for a university **Computational Intelligence** course. It investigates how different swarm dynamics influence the search for cluster centroids and provides a controlled, visual environment for comparing PSO-based clustering with a K-Means baseline.

The application is designed as an experimentation tool rather than a packaged machine-learning library: users can change the clustering problem, optimization parameters, and evaluation objective, then inspect the resulting partitions and centroids immediately. It emphasizes population-based optimization, swarm intelligence, and unsupervised learning.

## Learning Objectives

This project was used to apply and investigate:

1. Particle Swarm Optimization as a population-based metaheuristic.
2. Continuous centroid optimization for an unsupervised clustering problem.
3. The exploration-exploitation trade-off in stochastic optimization.
4. Global and neighborhood information sharing in swarm topologies.
5. The effect of PSO hyperparameters on observed clustering results.
6. Quantitative evaluation using complementary cluster-quality measures.
7. Interactive experimentation and comparison under a shared preprocessing pipeline.

## Why This Project Matters

Clustering with K-Means is efficient and widely used, but its solution depends on initialization and its local search can settle in a local optimum. This project frames clustering as a continuous optimization problem:

- A particle represents `k` candidate centroids flattened into one position vector.
- The fitness function assigns every sample to its nearest candidate centroid.
- Fitness is the sum of squared distances between samples and their assigned centroids.
- PSO variants explore this search space using different information-sharing, inertia, velocity-control, and mutation strategies.

The application makes those design choices observable. It is useful for studying the exploration-exploitation trade-off, swarm topology, parameter sensitivity, and the relationship between an optimization objective and an external clustering-quality metric.

## Features

- Interactive Streamlit interface for running clustering experiments.
- K-Means baseline and six PSO implementations.
- User-controlled cluster count, numeric feature selection, and random seed.
- Manual controls for swarm size, iterations, inertia, acceleration coefficients, velocity limits, and mutation behavior.
- Automatic Cartesian parameter search for all seven algorithms, with selection by either inertia or silhouette score.
- Comparison tables and charts for manual and auto-tuned runs.
- 2D feature-space plots with centroid markers.
- Interactive Plotly 3D plots when at least three features are selected.
- PCA projections for datasets with more than two selected features.
- Original-scale centroid reporting for easier interpretation.
- Support for the included CSV dataset and user-uploaded CSV files.

## Algorithms Implemented

| Method | Implementation focus |
| --- | --- |
| **K-Means** | Scikit-learn baseline that minimizes within-cluster sum of squares. |
| **Global-Best PSO** | Each particle is influenced by its personal best and the best solution found by the full swarm. |
| **Local-Best PSO** | Uses a ring neighborhood so particles follow a neighborhood best rather than a single global attractor. |
| **Linear-Inertia PSO** | Decreases inertia from `w_max` to `w_min` across iterations to shift from exploration toward exploitation. |
| **Constriction-Factor PSO** | Applies a constriction coefficient computed from `c1 + c2` to regulate velocity updates. |
| **Velocity-Clamped PSO** | Limits each velocity component using a configurable fraction of the feature range. |
| **PSO with Gaussian Mutation** | Adds Gaussian perturbations to particles with configurable probability and scale. |

All PSO implementations initialize centroid candidates within the observed feature bounds, update personal/global or neighborhood bests, clip positions to those bounds, and return labels and centroids for evaluation.

## Evaluation

Experiments report two complementary measures calculated on standardized feature values:

- **Inertia**: the sum of squared distances from each sample to its assigned centroid. Lower values indicate tighter clusters, although inertia generally decreases as `k` increases.
- **Silhouette score**: compares within-cluster cohesion with separation from other clusters. Values range from `-1` to `1`; higher values generally indicate better-separated clusters.

The two metrics can disagree. The best algorithm should therefore be selected according to the goal of the experiment rather than by treating one metric as universally decisive.

## Auto-Parameter Tuning

The Auto-Parameter Tuning tab evaluates predefined parameter grids for K-Means and each PSO variant. You can choose:

- the number of runs per parameter combination;
- the number of values sampled for each parameter; and
- either silhouette score or inertia as the selection criterion.

The application reports the best observed parameterized run for each algorithm, displays its parameters and metrics, and compares the selected configurations. Because the search is Cartesian, runtime grows quickly as the number of variations increases. Results are stochastic for PSO, so comparisons should be repeated with multiple seeds when drawing stronger conclusions.

The tuning implementation tests the following parameter families:

- particle count (`swarm_size`);
- iteration budget (`max_iters`);
- inertia and acceleration coefficients;
- linear-inertia bounds (`w_max`, `w_min`);
- velocity limit fraction (`vmax_frac`); and
- mutation probability and scale (`mutation_prob`, `mutation_scale`).

This is a predefined grid search, not Bayesian optimization or a learned hyperparameter model. K-Means tuning varies its `random_state` values, while PSO runs use the NumPy random generator used by the optimizer.

## Dataset

The repository includes `Mall_Customers.csv`, a customer-segmentation dataset with the following columns:

- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

The application clusters numeric features and standardizes the selected columns with scikit-learn's `StandardScaler` before optimization. `CustomerID` is not a meaningful behavioral feature and should normally be excluded from an experiment. The default experiment is intended to use `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.

You can also upload another CSV file through the Streamlit sidebar. Uploaded data should contain at least two numeric columns; select the columns that should define similarity for the clustering task.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone <your-repository-url>
cd ci-project-main
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

From the repository root:

```bash
streamlit run app/streamlit_app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

### Default dataset filename note

The checked-in dataset is named `Mall_Customers.csv`, while the current application code looks for `Mall_Customers.csv` when the default-dataset option is enabled. To run the default dataset without changing the Python code, create a copy with the expected name:

**Windows PowerShell**

```powershell
Copy-Item .\Mall_Customers.csv ".\Mall_Customers.csv"
```

Alternatively, disable the default-dataset option and upload `Mall_Customers.csv` through the interface.

## Suggested Experiment Workflow

1. Start with the default features and choose a value of `k`.
2. Run K-Means to establish a baseline inertia and silhouette score.
3. Run each PSO variant using the same dataset, feature set, `k`, and seed.
4. Compare both inertia and silhouette score rather than relying on one measure.
5. Use Auto-Parameter Tuning to explore predefined parameter combinations.
6. Repeat promising configurations with several random seeds before making a conclusion about algorithm quality.
7. Inspect the centroid table and visualizations to connect the numerical metrics with interpretable cluster structure.

## Experimental Questions

The application supports experiments such as:

- How do PSO variants compare with K-Means for the same centroid-based clustering task?
- How does ring-neighborhood information sharing change the observed search behavior?
- How do decreasing inertia, constriction, and velocity limits affect the exploration-exploitation balance?
- Does Gaussian mutation produce different cluster-quality scores under the same budget?
- How sensitive are the variants to swarm size, iteration count, and acceleration coefficients?
- Do the configurations selected by silhouette score differ from those selected by inertia?

These are empirical questions. The application reports the results of the selected runs; it does not claim that one variant is universally superior.

## Project Structure

```text
.
├── Mall_Customers.csv          # Included sample dataset
├── requirements.txt            # Python dependencies
└── app/
    ├── streamlit_app.py        # Interactive experiment interface
    ├── functions2.py           # PSO implementations and clustering helpers
    ├── data.py                 # Standalone sample-data preprocessing helper
    └── CI PSO Variations.ipynb # Course notebook and exploratory work
```

## Technical Notes and Limitations

- The current PSO functions use NumPy's global random generator; the Streamlit interface seeds it from the selected random-seed control before running an experiment.
- PSO and K-Means results can vary with initialization. A single run is not sufficient evidence of statistical superiority.
- Auto-tuning selects the best observed run, not the mean or variance across runs. For a formal study, record repeated-run results and report aggregate statistics.
- The application expects numeric feature columns for clustering; categorical columns such as `Gender` are not encoded automatically.
- The repository does not currently include an automated test suite or benchmark results. Reported performance should therefore be generated from the interface for the chosen dataset, features, seed, and parameters.

## Academic Context

This project demonstrates practical understanding of:

- population-based and swarm-intelligence optimization;
- continuous encoding of a discrete clustering assignment problem;
- neighborhood topology and information flow in PSO;
- adaptive inertia, constriction, velocity clamping, and mutation;
- data standardization and dimensionality reduction for analysis;
- objective-function design and multi-criterion evaluation; and
- empirical comparison of stochastic optimization methods.

The code is intentionally structured so that a new PSO variation can be implemented as a separate optimizer, connected to the Streamlit interface, and evaluated with the same clustering pipeline.

## Technologies

- Python
- Streamlit for the interactive application
- NumPy for numerical computation and random sampling
- Pandas for CSV loading and tabular result display
- scikit-learn for scaling, K-Means, PCA, silhouette score, and nearest-centroid distance calculations
- Matplotlib and Seaborn for static plots and correlation analysis
- Plotly for interactive 3D cluster visualization

## Potential Extensions

The current repository provides a foundation for further experiments, but these capabilities are not implemented yet:

- convergence-history recording and plotting;
- repeated-run summary statistics and confidence intervals;
- statistical significance testing across seeds;
- additional cluster-validity indices;
- hybrid PSO/K-Means refinement;
- comparisons with Fuzzy C-Means, Genetic Algorithms, or Differential Evolution;
- multi-objective clustering; and
- benchmark datasets and larger-scale performance experiments.

## License

This project was developed for academic and educational purposes.
