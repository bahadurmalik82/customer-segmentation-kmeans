# Customer Segmentation with K-Means — Including a Custom Implementation

Segmenting 200 Mall Customers with K-Means, and proving I understand the algorithm by writing it from scratch and checking it against sklearn — number for number.

## The problem

- **Dataset:** Mall Customers (`Mall_Customers.csv`, 200 rows, 0 missing, 0 duplicates)
- **Features:** Age (mean 38.9), Annual Income (mean $60.6k), Spending Score (mean 50.2). Gender: 112 female, 88 male.
- **Goal:** find customer segments a business can actually act on — not just coloured dots on a plot.

## Results

### Choosing K — Elbow + Silhouette (Income + Spending, unscaled)

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 181,363.60 | 0.2969 |
| 3 | 106,348.37 | 0.4676 |
| 4 | 73,679.79 | 0.4932 |
| **5** | **44,448.46** | **0.5539** |
| 6 | 37,233.81 | 0.5398 |
| 7 | 30,241.34 | 0.5288 |

**K = 5** — elbow bend + highest silhouette (0.5539). With Age added and features scaled, K = 5 gives silhouette 0.4166, and K = 4 gives 0.4040 — scaling changes the answer, which is itself a finding (see below).

### The 5 segments (K = 5, Income + Spending)

| Segment | Size | Mean age | Mean income (k$) | Mean spending | % female | Business name | Action |
|---|---:|---:|---:|---:|---:|---|---|
| Cluster 2 | 22 | 25.3 | 25.7 | 79.4 | 59.1 | Young big spenders | New-arrival drops, social campaigns |
| Cluster 4 | 23 | 45.2 | 26.3 | 20.9 | 60.9 | Careful savers | Value bundles, durability messaging |
| Cluster 0 | 81 | 42.7 | 55.3 | 49.5 | 59.3 | Steady shoppers | Loyalty programme |
| Cluster 1 | 39 | 32.7 | 86.5 | 82.1 | 53.8 | Premium whales | VIP invites, early access |
| Cluster 3 | 35 | 41.1 | 88.2 | 17.1 | 45.7 | High income, low spend | Premium quality pitch, not discounts |

Only 2 of 200 customers have a negative silhouette — the clusters are clean. Adding Age (3-feature, K = 4) re-splits the data (ARI 0.4706 vs the 2-feature solution), so Age changes the story rather than confirming it. Gender does not separate the segments: % female is 46–61% in every cluster.

### Custom K-Means vs sklearn — verified, not claimed

I implemented K-Means from scratch (`my_kmeans` in the notebook) and checked it point by point:

- Customer 1, hand-computed silhouette: a = 24.64, b = 42.27, s = **0.4170** — sklearn: **0.4170**. Exact match.
- 1-D practice set `[1, 3, 5, 7, 11, 13, 15]`, 2 centroids: final centroids `[4.0, 13.0]`, 3 iterations, per-point silhouettes match sklearn (0.6667 / 0.3333 / 0.5714).

### Initialisation matters — 30 seeds

| Setting | Distinct final solutions | % of runs finding the best inertia (44,448.46) |
|---|---:|---:|
| random, n_init=1 | 14 | 33.3% |
| k-means++, n_init=1 | 6 | 60.0% |
| k-means++, n_init=10 | **1** | **100%** |

Random init with one restart finds the optimum only a third of the time. `n_init=10` is not a default to ignore.

### Scaling matters

On the Wine dataset in the notebook, scaling flips the verdict depending on which metric you trust: unscaled — silhouette 0.5711 but ARI vs the true cultivars only 0.3711; scaled — silhouette falls to 0.2849 but ARI jumps to 0.8975. Silhouette alone would tell you scaling made things worse; agreement with ground truth says the opposite. K-Means uses Euclidean distance, so unscaled features let the largest-scale feature dominate — always state whether you scaled, and judge with more than one metric.

### Outliers break K-Means — measured

One extreme customer (Income 400, Spending 99) added to the 200:

- Its distance to the nearest centroid: **313.92** vs a mean + 3σ threshold of **34.80** — flagged as an anomaly.
- Without repair it hijacks a whole cluster (sizes become [100, 1, 23, 39, 38]) and agreement with the original segmentation falls to ARI 0.7532.
- After distance-based repair, cluster sizes are [80, 34, 23, 37, 22] and ARI recovers to **0.9845**. 4 real customers were also flagged by the same rule — inspect them before deleting anyone.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook KMeans_Clustering.ipynb
```

`Mall_Customers.csv` is committed in this repo. The notebook writes the final segmentation to `customer_segments.csv`.

## Structure

```
.
├── KMeans_Clustering.ipynb   # full analysis: EDA, Elbow/Silhouette, custom K-Means,
│                             #   init stability, scaling, outlier repair, segment profiles
├── Mall_Customers.csv        # input (200 rows)
├── customer_segments.csv     # output: customers with cluster labels
├── README.md
└── requirements.txt
```

## Limitations

- 200 customers from one mall — segments may not generalise to another store or country.
- K-Means assumes spherical, equally sized clusters; DBSCAN / Gaussian mixtures were not compared here.
- Segment names and actions are business interpretations of the profile table, not tested interventions.
- The custom implementation lives in the notebook — the next step is extracting it to `src/kmeans_custom.py` with unit tests.

## Author

Bahadur Ali Meer Mohammad — BS Computer Science (AI), Sukkur IBA University.
Machine Learning Engineer / Agent Builder (LLM Engineer). LinkedIn: https://www.linkedin.com/in/bahadur-ali82/
