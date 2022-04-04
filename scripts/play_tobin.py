#%%
from lib2to3.pgen2.grammar import Grammar
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

n = len(right_adj)
p = (p1 + p2) / 2
rho = 0.0


def obj_func(A, B, perm):
    PBPT = B[perm[: len(A)]][:, perm[: len(A)]]
    return np.linalg.norm(A - PBPT, ord="fro") ** 2


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
            score = obj_func(A, B, perm_inds)

            rows.append({"method": method, "score": score, "data": data, "sim": sim})

results = pd.DataFrame(rows)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
from pkg.plot import set_theme

set_theme()
fig, axs = plt.subplots(2, 1, figsize=(8, 12))
ax = axs[0]
sns.histplot(
    data=results[results["data"] == "er"],
    x="score",
    hue="method",
    bins=50,
    kde=True,
    ax=ax,
)
ax.set_xlabel("Network difference magnitude")
ax.set(ylabel="", yticks=[])
ax.spines["left"].set_visible(False)

ax = axs[1]
sns.histplot(
    data=results[results["data"] == "true"],
    x="score",
    hue="method",
    bins=50,
    kde=True,
    ax=ax,
)
ax.set_xlabel("Network difference magnitude")
ax.set(ylabel="", yticks=[])
ax.spines["left"].set_visible(False)


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
