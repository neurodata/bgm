#%% [markdown]
# # Maggot connectome subset
#%%
import datetime
import logging
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pymaid
import seaborn as sns
from graspologic.match import graph_match
from graspologic.plot import adjplot
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from pkg.data import DATA_PATH, load_split_connectome
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import (
    dashes,
    matched_stripplot,
    method_palette,
    rgb2hex,
    set_theme,
    simple_plot_neurons,
    subgraph_palette,
)
from sklearn.model_selection import KFold, train_test_split
from tqdm.autonotebook import tqdm

FILENAME = "connectome_seeded"

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
set_theme()
rng = np.random.default_rng(8888)


#%% [markdown]
# ## Start Catmaid instance on Virtual Fly Brain
#%%

pymaid.CatmaidInstance("https://l1em.catmaid.virtualflybrain.org/", None)
logging.getLogger("pymaid").setLevel(logging.WARNING)
pymaid.clear_cache()


#%%
def get_indicator_from_annotation(annot_name, filt=None):
    ids = pymaid.get_skids_by_annotation(annot_name.replace("*", "\*"))
    if filt is not None:
        name = filt(annot_name)
    else:
        name = annot_name
    indicator = pd.Series(
        index=ids, data=np.ones(len(ids), dtype=bool), name=name, dtype=bool
    )
    return indicator


annot_df = pymaid.get_annotated("papers")
series_ids = []

for annot_name in annot_df["name"]:
    print(annot_name)
    indicator = get_indicator_from_annotation(annot_name)
    if annot_name == "Imambocus et al":
        indicator.name = "Imambocus et al. 2022"
    series_ids.append(indicator)
nodes = pd.concat(series_ids, axis=1, ignore_index=False).fillna(False)


# adj_df = pymaid.adjacency_matrix(nodes.index.values)
# adj_df = pd.DataFrame(
#     data=adj_df.values.astype(int), index=adj_df.index, columns=adj_df.columns
# )

#%%
raw_path = DATA_PATH / "maggot"

paired_nodes = pd.read_csv(raw_path / "nodes.csv", index_col=0)

#%%


# %%
# get_indicator_from_annotation('soma left')

temp_meta = pd.read_csv("bgm/data/maggot/meta_data.csv", index_col=0)
temp_meta = temp_meta[temp_meta["hemisphere"] != "C"]

#%%
intersect_ids = nodes.index.intersection(temp_meta.index)
nodes = nodes.loc[intersect_ids]
nodes["hemisphere"] = temp_meta.loc[intersect_ids, "hemisphere"]

#%%
nodes["pair"] = np.nan
nodes.loc[paired_nodes.index, "pair"] = paired_nodes["pair"]
nodes["pair"] = nodes["pair"].astype("Int64")
nodes

#%%
nodes = nodes.sort_values("hemisphere")
nodes

#%%

adj_df = pymaid.adjacency_matrix(nodes.index.values)
adj_df = pd.DataFrame(
    data=adj_df.values.astype(int), index=adj_df.index, columns=adj_df.columns
)

#%%
left_nodes = nodes[nodes["hemisphere"] == "L"]
right_nodes = nodes[nodes["hemisphere"] == "R"]

left_node_ids = left_nodes.index
right_node_ids = right_nodes.index

ll_adj = adj_df.reindex(index=left_node_ids, columns=left_node_ids).values
rr_adj = adj_df.reindex(index=right_node_ids, columns=right_node_ids).values
lr_adj = adj_df.reindex(index=left_node_ids, columns=right_node_ids).values
rl_adj = adj_df.reindex(index=right_node_ids, columns=left_node_ids).values

print(len(left_node_ids))
print(len(right_node_ids))


#%%
# indices_A, indices_B, score, misc = graph_match(
#     ll_adj,
#     rr_adj,
#     rng=rng,
#     verbose=1,
#     n_init=1,
# )

# left_nodes_sorted = left_nodes.iloc[indices_A]
# right_nodes_sorted = right_nodes.iloc[indices_B]

# equal_pairs = left_nodes_sorted["pair"].values == right_nodes_sorted["pair"].values
# real_pairs = (~left_nodes_sorted["pair"].isna().values) & (
#     ~right_nodes_sorted["pair"].isna().values
# )

# match_ratio = np.mean(equal_pairs[real_pairs])
# print(f"Match ratio = {match_ratio}")

#%%
def select_seeds(left_nodes, right_nodes, pairs="all"):
    left_nodes = left_nodes.copy()
    right_nodes = right_nodes.copy()
    left_nodes.index.name = "node_id"
    right_nodes.index.name = "node_id"
    left_nodes["pos_index"] = range(len(left_nodes))
    right_nodes["pos_index"] = range(len(right_nodes))

    left_paired_nodes = left_nodes[~left_nodes["pair"].isna()].copy()
    right_paired_nodes = right_nodes[~right_nodes["pair"].isna()].copy()
    if pairs == "all":
        pairs = np.intersect1d(
            left_paired_nodes["pair"],
            right_paired_nodes["pair"],
        )

    left_paired_nodes = left_paired_nodes.reset_index().set_index("pair")
    right_paired_nodes = right_paired_nodes.reset_index().set_index("pair")

    left_inds = left_paired_nodes.loc[pairs, "pos_index"]
    right_inds = right_paired_nodes.loc[pairs, "pos_index"]
    seeds = np.column_stack((left_inds, right_inds))
    return seeds


all_seeds = select_seeds(left_nodes, right_nodes)

# from sklearn.cros
indices = np.arange(len(all_seeds))


rerun = False
if rerun:
    rows = []
    kfold = KFold(shuffle=True)
    n_folds = 5
    n_seeds_range = [0, 100, 200, 300, 400]
    pbar = tqdm(total=n_folds * len(n_seeds_range) * 2)
    for fold, (indices_train, indices_test) in enumerate(kfold.split(indices)):

        # indices_train, indices_test = train_test_split(indices, test_size=0.2)

        test_seeds = all_seeds[indices_test]
        left_nodes_to_check = left_nodes.iloc[test_seeds[:, 0]].index

        rng.shuffle(indices_train)

        for n_seeds in n_seeds_range:
            selected_seeds = all_seeds[indices_train[:n_seeds]]

            for method in ["GM", "BGM"]:
                if method == "GM":
                    AB = None
                    BA = None
                else:
                    AB = lr_adj
                    BA = rl_adj
                indices_A, indices_B, score, misc = graph_match(
                    ll_adj,
                    rr_adj,
                    AB=AB,
                    BA=BA,
                    rng=rng,
                    n_init=1,
                    partial_match=selected_seeds,
                )

                left_nodes_sorted = left_nodes.iloc[indices_A]
                right_nodes_sorted = right_nodes.iloc[indices_B]

                correct = 0
                for left_node in left_nodes_to_check:
                    if left_node in left_nodes_sorted.index:
                        iloc = left_nodes_sorted.index.get_loc(left_node)
                        check = (
                            left_nodes_sorted.iloc[iloc]["pair"]
                            == right_nodes_sorted.iloc[iloc]["pair"]
                        )
                        if isinstance(check, np.bool_) and check:
                            correct += 1
                match_ratio_heldout = correct / len(indices_test)
                rows.append(
                    {
                        "match_ratio": match_ratio_heldout,
                        "n_seeds": n_seeds,
                        "method": method,
                        "fold": fold,
                    }
                )
                pbar.update(1)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH / "matching_results.csv")
else:
    results = pd.read_csv(OUT_PATH / "matching_results.csv", index_col=0)
#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(
    data=results,
    x="n_seeds",
    y="match_ratio",
    hue="method",
    palette=method_palette,
    style="method",
    dashes=dashes,
)
sns.move_legend(ax, loc="lower right", title="Method")
ax.set(ylabel="Matching accuracy", xlabel="Number of seeds", xticks=n_seeds_range)
gluefig("accuracy_by_seeds", fig)

#%%

currtime = time.time()
match_counts = np.zeros((ll_adj.shape[0], rr_adj.shape[0]))
n_init = 50
for i in tqdm(range(n_init)):
    indices_A, indices_B, score, misc = graph_match(
        ll_adj,
        rr_adj,
        AB=lr_adj,
        BA=rl_adj,
        rng=rng,
        n_init=1,
        partial_match=all_seeds,
    )
    match_counts[indices_A, indices_B] += 1

elapsed = time.time() - currtime

#%%
from scipy.optimize import linear_sum_assignment

row_counts = np.count_nonzero(match_counts, axis=1)
strong_match_indices = np.where(row_counts == 1)[0]
strong_match_indices = np.setdiff1d(strong_match_indices, all_seeds[:, 0])
indices_A, indices_B = linear_sum_assignment(match_counts, maximize=True)
p_matched = match_counts[indices_A, indices_B] / n_init
# for row in match_counts:
#     if np.count_nonzero(row) == 1:
#         pass
#     else:
#         print(np.count_nonzero(row))

#%%

left_nodes_sorted = left_nodes.iloc[indices_A].copy()
right_nodes_sorted = right_nodes.iloc[indices_B].copy()
left_nodes_sorted["p_matched"] = p_matched

#%%
equal_pairs = left_nodes_sorted["pair"].values == right_nodes_sorted["pair"].values
real_pairs = (~left_nodes_sorted["pair"].isna().values) & (
    ~right_nodes_sorted["pair"].isna().values
)

match_ratio = np.mean(equal_pairs[real_pairs])
print(f"Match ratio = {match_ratio}")

#%%
new_pairs = (
    left_nodes_sorted["pair"].isna().values & right_nodes_sorted["pair"].isna().values
)
new_left_nodes = left_nodes_sorted.loc[new_pairs]
new_right_nodes = right_nodes_sorted.loc[new_pairs]


#%%


def draw_box(ax, color="black"):
    # REF: https://github.com/matplotlib/matplotlib/blob/81e955935a26dae7048758f7b3dc3f1dc4c5de6c/lib/matplotlib/axes/_axes.py#L749
    xtrans = ax.get_xaxis_transform(which="grid")
    ytrans = ax.get_yaxis_transform(which="grid")
    xmin = -0.095
    xmax = 0.09
    ymin = -0.095
    ymax = 0.09
    trans = blended_transform_factory(xtrans, ytrans)
    kwargs = dict(transform=trans, clip_on=False, color=color)
    line = Line2D([xmin, xmin], [ymin, ymax], **kwargs)
    ax.add_line(line)
    line = Line2D([xmax, xmax], [ymin, ymax], **kwargs)
    ax.add_line(line)
    line = Line2D([xmin, xmax], [ymin, ymin], **kwargs)
    ax.add_line(line)
    line = Line2D([xmin, xmax], [ymax, ymax], **kwargs)
    ax.add_line(line)


colors = [subgraph_palette["LL"], subgraph_palette["RR"]]
colors = [rgb2hex(*color) for color in colors]

n_show = 15
n_cols = 5
n_rows = int(np.ceil(n_show / n_cols))
fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows), constrained_layout=True)
gs = plt.GridSpec(n_rows, n_cols, figure=fig, hspace=0, wspace=0)
axs_shape = (n_rows, n_cols)
indices = rng.choice(n_show, size=n_show, replace=False)
for i, index in enumerate(indices):
    ax_inds = np.unravel_index(i, axs_shape)
    ax = fig.add_subplot(gs[ax_inds], projection="3d")
    neurons = [new_left_nodes.index[index], new_right_nodes.index[index]]
    palette = dict(zip(neurons, colors))
    simple_plot_neurons(
        neurons,
        palette=palette,
        ax=ax,
        force_bounds=False,
        autoscale=True,
        soma=False,
        dist=3.5,
        lw=1.5,
        elev=0,
        azim=0,
    )
    # ax.set_xlim3d((-4500, 110000))
    # ax.set_ylim3d((-4500, 110000))
    draw_box(ax, color="lightgrey")

gluefig("example_matched_morphologies", fig=fig)

# %%

n_show = 7
n_cols = n_show
n_rows = 3
views = (dict(elev=-90, azim=90), dict(elev=0, azim=0), dict(elev=45, azim=45))
fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows), constrained_layout=True)
gs = plt.GridSpec(n_rows, n_cols, figure=fig, hspace=0, wspace=0)
axs_shape = (n_rows, n_cols)

# only select the best matches
left_choice_index = new_left_nodes[new_left_nodes["p_matched"] == 1].index
# pick some random ones
choice_left_ids = rng.choice(left_choice_index, size=n_show, replace=False)

for j, choice_left_id in enumerate(choice_left_ids):
    choice_iloc = new_left_nodes.index.get_loc(choice_left_id)
    choice_right_id = new_right_nodes.index[choice_iloc]
    neurons = [choice_left_id, choice_right_id]
    palette = dict(zip(neurons, colors))
    for i, view in enumerate(views):
        # ax_inds = np.unravel_index(i, axs_shape)
        ax = fig.add_subplot(gs[(i, j)], projection="3d")
        simple_plot_neurons(
            neurons,
            palette=palette,
            ax=ax,
            force_bounds=False,
            autoscale=True,
            soma=False,
            dist=3,
            lw=1.5,
            **view,
        )
        # ax.set_xlim3d((-4500, 110000))
        # ax.set_ylim3d((-4500, 110000))
        draw_box(ax, color="lightgrey")
gluefig("example_matched_morphologies_3view", fig=fig)
