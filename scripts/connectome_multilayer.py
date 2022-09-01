#%% [markdown]
# # Connectome data

#%%
import datetime

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.data import load_split_connectome
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import matched_stripplot, method_palette, set_theme
from pkg.utils import get_hemisphere_indices
from scipy.stats import wilcoxon
from tqdm.autonotebook import tqdm
from graspologic.match import graph_match


FILENAME = "connectome_multilayer"

DISPLAY_FIGS = True

OUT_PATH = OUT_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
rng = np.random.default_rng(8888)

#%% [markdown]
# ## Load processed data, run matching experiment
#%%

from pkg.data import DATA_PATH
import networkx as nx


def load_multilayer_connectome(dataset, weights=True, layers=["chem", "elec"]):
    folder = DATA_PATH / "processed_full"
    if weights:
        data = (("weight", int),)
    else:
        data = False
    if dataset in [
        "herm",
        "male",
        "specimen_107",
        "specimen_148",
    ]:
        nodetype = str
    elif dataset in ["maggot", "maggot_subset"]:
        nodetype = int

    nodes = pd.read_csv(folder / f"{dataset}_nodes.csv", index_col=0)

    adjs = []
    for layer in layers:
        if layer == "elec":
            create_using = nx.Graph
        else:
            create_using = nx.DiGraph
        g = nx.read_edgelist(
            folder / f"{dataset}_{layer}_edgelist.csv",
            create_using=create_using,
            delimiter=",",
            nodetype=nodetype,
            data=data,
        )
        for node in nodes.index:
            g.add_node(node)
        adj = nx.to_numpy_array(g, nodelist=nodes.index)
        adjs.append(adj)

    return adjs, nodes


def select_subgraph(adjs, source_inds, target_inds):
    new_adjs = []
    for adj in adjs:
        new_adjs.append(adj[source_inds][:, target_inds])
    return new_adjs


RERUN_SIMS = False
datasets = ["male", "herm"]

n_sims = 10
glue("n_initializations", n_sims)

if RERUN_SIMS:
    rows = []
    pbar = tqdm(total=len(datasets) * 3 * 2 * n_sims)
    for dataset in datasets:
        adjs, nodes = load_multilayer_connectome(dataset)
        n_nodes = len(nodes)

        left_inds, right_inds = get_hemisphere_indices(nodes)

        A = select_subgraph(adjs, left_inds, left_inds)
        B = select_subgraph(adjs, right_inds, right_inds)
        AB = select_subgraph(adjs, left_inds, right_inds)
        BA = select_subgraph(adjs, right_inds, left_inds)

        n_side = len(left_inds)
        for layer in ["chem", "elec", "both"]:
            if layer == "chem":
                layer_A = A[0]
                layer_B = B[0]
                layer_AB = AB[0]
                layer_BA = BA[0]
            elif layer == "elec":
                layer_A = A[1]
                layer_B = B[1]
                layer_AB = AB[1]
                layer_BA = BA[1]
            else:
                layer_A = A
                layer_B = B
                layer_AB = AB
                layer_BA = BA

            for method in ["GM", "BGM"]:
                if method == "GM":
                    this_AB = None
                    this_BA = None
                elif method == "BGM":
                    this_AB = layer_AB
                    this_BA = layer_BA

                this_A = layer_A
                this_B = layer_B

                for sim in range(n_sims):

                    currtime = time.time()
                    indices_A, indices_B, score, misc = graph_match(
                        this_A, this_B, AB=this_AB, BA=this_BA, rng=sim
                    )
                    elapsed = time.time() - currtime

                    match_ratio = (indices_B == np.arange(n_side)).mean()
                    rows.append(
                        {
                            "match_ratio": match_ratio,
                            "sim": sim,
                            "method": method,
                            "layer": layer,
                            "elapsed": elapsed,
                            "converged": misc[0]["converged"],
                            "n_iter": misc[0]["n_iter"],
                            "score": score,
                            "dataset": dataset,
                            "chem": (layer == "chem") or (layer == "both"),
                            "elec": (layer == "elec") or (layer == "both"),
                            "ipsi": True,
                            "contra": (method == "BGM"),
                        }
                    )
                    pbar.update(1)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH / "match_results.csv")
else:
    results = pd.read_csv(OUT_PATH / "match_results.csv", index_col=0)

# %%


def draw_significance(ax, pvalue, x, xdist, y=1.02, ydist=0.03):
    if pvalue < 0.0005:
        text = "***"
    elif pvalue < 0.005:
        text = "**"
    elif pvalue < 0.05:
        text = "*"
    else:
        text = ""
    if text != "":
        ax.plot(
            [x - xdist, x - xdist, x + xdist, x + xdist],
            [y, y + ydist, y + ydist, y],
            color="dimgrey",
            clip_on=False,
        )
        ax.text(x, y, text, ha="center", va="bottom", fontsize="large")


grouper = results.groupby(["chem", "elec", "ipsi", "contra"])
sort_order = grouper["match_ratio"].mean().sort_values()
sort_order = sort_order.to_frame()
sort_order["sorter"] = np.arange(len(sort_order))


results["sorter"] = np.nan
for key, inds in grouper.groups.items():
    results.loc[inds, "sorter"] = sort_order.loc[key, "sorter"]


results = results.sort_values("sorter")

from giskard.plot import upset_catplot
from scipy.stats import mannwhitneyu

set_theme()

fig, axs = plt.subplots(
    1,
    2,
    figsize=(12, 6),
    sharey=False,
    constrained_layout=True,
    gridspec_kw=dict(wspace=0.12),
)
category_name_map = {
    "chem": "Chemical",
    "elec": "Electrical",
    "ipsi": "Ipsilateral",
    "contra": "Contralateral",
}
nice_dataset_map = {
    "herm": "C. elegans hermaphrodite",
    "male": "C. elegans male",
}
for i, (dataset, data_results) in enumerate(results.groupby("dataset", sort=False)):
    ax = axs.flat[i]

    uc = upset_catplot(
        data=data_results,
        x=["chem", "elec", "ipsi", "contra"],
        y="match_ratio",
        kind="strip",
        ax=ax,
        hue="method",
        palette=method_palette,
        jitter=0.2,
        estimator_labels=True,
        estimator=np.mean,
    )
    uc.set_upset_ticklabels(category_name_map)
    if i == 0:
        sns.move_legend(ax, loc="upper left", title="Method", frameon=True)
    else:
        ax.get_legend().remove()
    ax.set(ylabel="Matching accuracy", title=nice_dataset_map[dataset])
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    for i, (combo, combo_results) in enumerate(
        data_results.groupby("layer", sort=False)
    ):
        ratios = []
        for method, method_results in combo_results.groupby("method"):
            ratios.append(method_results["match_ratio"])
        stat, pvalue = mannwhitneyu(*ratios)
        print(dataset, combo, pvalue)

        # draw_significance(
        #     ax,
        #     pvalue,
        #     x=i * 2 + 0.5,
        #     xdist=0.5,
        #     y=combo_results["match_ratio"].max() + 0.03,
        #     # ydist=0.05,
        # )

# for i, (group_label, group_results) in enumerate(results.groupby(["dataset", "layer"])):
#     ratios = []
#     for method, method_results in group_results.groupby("method"):
#         ratios.append(method_results["match_ratio"])
#     stat, pvalue = mannwhitneyu(*ratios)
#     print(group_label, pvalue)
#     print()
#     if group_label[0] == "male":
#         ax = axs[1]
#     else:
#         ax = axs[0]
# draw_significance(ax, pvalue, x=i, xdist=0.5, y=1.02)


gluefig("accuracy_upsetplot", fig)
