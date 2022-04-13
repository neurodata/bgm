#%%
import pandas as pd

filename = "bgm/data/elife-24838-fig1-figsupp4-data1-v1.csv"
adj_df = pd.read_csv(filename, index_col=0)
adj_df
# %%
from graspologic.plot import heatmap

heatmap(adj_df.values, transform="binarize")

#%%
heatmap(adj_df.values[5:, 5:], transform="binarize")

#%%
orn_adj_df = adj_df.iloc[5:, 5:]

#%%
node_ids = orn_adj_df.index
nodes = pd.DataFrame(index=node_ids)
sides = [n.split(" ")[0] for n in node_ids]
sides = list(map({"left": "L", "right": "R"}.get, sides))
nodes["hemisphere"] = sides
nodes["_inds"] = range(len(nodes))
left_inds = nodes[nodes["hemisphere"] == "L"]["_inds"]
right_inds = nodes[nodes["hemisphere"] == "R"]["_inds"]

#%%
from graspologic.match import GraphMatch
import numpy as np
from graspologic.utils import binarize

adj = binarize(orn_adj_df.values)

left_adj = adj[np.ix_(left_inds, left_inds)]
right_adj = adj[np.ix_(right_inds, right_inds)]


def compute_p(A):
    return np.count_nonzero(A) / (A.size - A.shape[0])


p1 = compute_p(right_adj)
p2 = compute_p(left_adj)

from graspologic.simulations import er_corr
from scipy.stats import pearsonr

n = len(right_adj)
p = (p1 + p2) / 2
rho = 0.0


def obj_func(A, B, perm):
    PBPT = B[perm[: len(A)]][:, perm[: len(A)]]
    return np.linalg.norm(A - PBPT, ord="fro") ** 2, PBPT


def ravel(A):
    triu_indices = np.triu_indices_from(A, k=1)
    tril_indices = np.tril_indices_from(A, k=-1)
    return np.concatenate((A[triu_indices], A[tril_indices]))


def compute_density(adjacency, loops=False):
    if not loops:
        triu_inds = np.triu_indices_from(adjacency, k=1)
        tril_inds = np.tril_indices_from(adjacency, k=-1)
        n_edges = np.count_nonzero(adjacency[triu_inds]) + np.count_nonzero(
            adjacency[tril_inds]
        )
    else:
        n_edges = np.count_nonzero(adjacency)
    n_nodes = adjacency.shape[0]
    n_possible = n_nodes**2
    if not loops:
        n_possible -= n_nodes
    return n_edges / n_possible


def compute_alignment_strength(A, B, perm=None):
    n = A.shape[0]
    if perm is not None:
        B_perm = B[perm][:, perm]
    else:
        B_perm = B
    n_disagreements = np.count_nonzero(A - B_perm)
    p_disagreements = n_disagreements / (n**2 - n)
    densityA = compute_density(A)
    densityB = compute_density(B)
    denominator = densityA * (1 - densityB) + densityB * (1 - densityA)
    alignment_strength = 1 - p_disagreements / denominator
    return alignment_strength


#


#%%
ravel(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

#%%

rng = np.random.default_rng()
n_sims = 1000
rows = []
for data in ["true", "er"]:
    if data == "true":
        A = right_adj
        B = left_adj
    elif data == "er":
        A, B = er_corr(n, p, rho, directed=True, loops=False)
    for sim in range(n_sims):
        for method in ["random", "gm"]:
            if method == "random":
                n = len(A)
                perm_inds = rng.permutation(n)
            elif method == "gm":
                gm = GraphMatch()
                gm.fit(A, B)
                perm_inds = gm.perm_inds_
            score, B_perm = obj_func(A, B, perm_inds)

            pearson_stat, pearson_pvalues = pearsonr(ravel(A), ravel(B_perm))

            alignment = compute_alignment_strength(A, B_perm)

            rows.append(
                {
                    "method": method,
                    "score": score,
                    "data": data,
                    "sim": sim,
                    "pearson_stat": pearson_stat,
                    "pearson_pvalues": pearson_pvalues,
                    "alignment": alignment,
                }
            )

results = pd.DataFrame(rows)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
from pkg.plot import set_theme

set_theme()
fig, axs = plt.subplots(
    2, 1, figsize=(8, 6), sharex=True, gridspec_kw=dict(hspace=0.01)
)
ax = axs[1]
sns.kdeplot(
    data=results[results["data"] == "er"],
    x="score",
    hue="method",
    # bins=50,
    # kde=True,
    fill=True,
    ax=ax,
    legend=False,
)
ax.set_xlabel("Network difference magnitude")
ax.set(ylabel="", yticks=[])
ax.set_ylabel("Independent\nER\nsimulation\n", rotation=0, ha="right", va="center")
ax.spines["left"].set_visible(False)

ax = axs[0]
sns.kdeplot(
    data=results[results["data"] == "true"],
    x="score",
    hue="method",
    # bins=50,
    # kde=True,
    fill=True,
    ax=ax,
    legend=True,
)
ax.set_xlabel("Network difference magnitude")
ax.set(ylabel="", yticks=[])
ax.set_ylabel("Observed\ndata", rotation=0, ha="right", va="center")
ax.spines["left"].set_visible(False)
sns.move_legend(ax, "upper right", title="Matching")

#%%

from giskard.plot import subuniformity_plot

x = results[(results["data"] == "er") & (results["method"] == "random")][
    "pearson_pvalues"
]
subuniformity_plot(x)
#%%

er_results = results[results["data"] == "er"]
true_results = results[results["data"] == "true"]


def compute_match_score_ratios(results):
    rand_results = results[results["method"] == "random"]
    gm_results = results[results["method"] == "gm"]
    ratios = gm_results["score"].values / rand_results["score"].values
    return ratios


true_ratios = compute_match_score_ratios(true_results)
er_ratios = compute_match_score_ratios(er_results)

obs_to_er_ratio = true_ratios.mean() / er_ratios.mean()


#%%
from graspologic.utils import is_loopless

is_loopless(adj)

#%% compute the alignment strength metric


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.kdeplot(
    data=results[results["method"] == "gm"],
    x="alignment",
    hue="data",
    ax=ax,
    fill=True,
)
ax.set(ylabel="", yticks=[], xlabel="Alignment strength")
ax.spines["left"].set_visible(False)
