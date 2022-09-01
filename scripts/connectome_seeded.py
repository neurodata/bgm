#%% [markdown]
# # Maggot connectome subset with seeds
#%%
import datetime
import time
import logging
import pymaid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.match import graph_match
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from pkg.data import load_semipaired_connectome
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import (
    dashes,
    method_palette,
    rgb2hex,
    set_theme,
    simple_plot_neurons,
    subgraph_palette,
)
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
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


pymaid.CatmaidInstance("https://l1em.catmaid.virtualflybrain.org/", None)
logging.getLogger("pymaid").setLevel(logging.WARNING)
pymaid.clear_cache()


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)

#%%

adj, nodes = load_semipaired_connectome("maggot_subset")
adj_df = pd.DataFrame(data=adj, index=nodes.index, columns=nodes.index)

#%%
left_nodes = nodes[nodes["hemisphere"] == "L"]
right_nodes = nodes[nodes["hemisphere"] == "R"]

left_node_ids = left_nodes.index
right_node_ids = right_nodes.index

ll_adj = adj_df.reindex(index=left_node_ids, columns=left_node_ids).values
rr_adj = adj_df.reindex(index=right_node_ids, columns=right_node_ids).values
lr_adj = adj_df.reindex(index=left_node_ids, columns=right_node_ids).values
rl_adj = adj_df.reindex(index=right_node_ids, columns=left_node_ids).values

n_left = len(left_node_ids)
n_right = len(right_node_ids)

glue("n_left", n_left)
glue("n_right", n_right)

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

indices = np.arange(len(all_seeds))
n_seeds_range = [0, 100, 200, 300, 400]

n_folds = 5
glue("n_folds", n_folds)

rerun = False
if rerun:
    rows = []
    kfold = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=rng.integers(np.iinfo(np.uint32).max),
    )

    pbar = tqdm(total=n_folds * len(n_seeds_range) * 2)
    for fold, (indices_train, indices_test) in enumerate(kfold.split(indices)):

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
set_theme(font_scale=1)
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
sns.move_legend(ax, loc="lower right", title="Method", frameon=True)
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.set(ylabel="Matching accuracy", xlabel="Number of seeds", xticks=n_seeds_range)
gluefig("accuracy_by_seeds", fig)

#%%

stat_rows = []
for n_seeds, seed_results in results.groupby("n_seeds"):
    ratios = []
    for method, method_results in seed_results.groupby("method"):
        ratios.append(method_results["match_ratio"])

    # believe this should be wilcoxon, paired across folds
    stat, pvalue = wilcoxon(*ratios, method="exact", zero_method="wilcox")
    # stat, pvalue = mannwhitneyu(*ratios)

    stat_rows.append({"n_seeds": n_seeds, "pvalue": pvalue, "stat": stat})
stat_results = pd.DataFrame(stat_rows)
stat_results.to_csv(OUT_PATH / "seeded_stat_results")
stat_results

#%%
for n_seeds, seed_results in results.groupby("n_seeds"):
    ratios = []
    for method, method_results in seed_results.groupby("method"):
        ratios.append(method_results["match_ratio"])
    print(ratios[1].values - ratios[0].values)

#%%
n_init = 100
glue("full_seed_n_init", n_init)

rng = np.random.default_rng(9999)

rerun = False
if rerun:
    match_probs = np.zeros((ll_adj.shape[0], rr_adj.shape[0]))
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
        match_probs[indices_A, indices_B] += 1 / n_init

    match_probs_df = pd.DataFrame(
        data=match_probs, index=left_node_ids, columns=right_node_ids
    )
    match_probs_df.to_csv(OUT_PATH / "match_probs.csv")
else:
    match_probs_df = pd.read_csv(OUT_PATH / "match_probs.csv", index_col=0)
    match_probs = match_probs_df.values


#%%

# from the full set of matching probabilities, get the most likely for each
indices_A, indices_B = linear_sum_assignment(match_probs, maximize=True)
# these are the matching probabilities for the final matches (if we have to choose one)
p_matched = match_probs[indices_A, indices_B]

# resort data accordingly
left_nodes_sorted = left_nodes.iloc[indices_A].copy()
right_nodes_sorted = right_nodes.iloc[indices_B].copy()
left_nodes_sorted["p_matched"] = p_matched

# sanity check - this should always be 1
equal_pairs = left_nodes_sorted["pair"].values == right_nodes_sorted["pair"].values
real_pairs = (~left_nodes_sorted["pair"].isna().values) & (
    ~right_nodes_sorted["pair"].isna().values
)
match_ratio = np.mean(equal_pairs[real_pairs])
print(f"Match ratio = {match_ratio}")

#%%
# sub select the new pairs only (not seeds)
new_pairs = (
    left_nodes_sorted["pair"].isna().values & right_nodes_sorted["pair"].isna().values
)

# make a dataframe for the new matches
new_left_nodes = left_nodes_sorted.loc[new_pairs].copy()
new_left_nodes.index.name = "skid_left"
new_right_nodes = right_nodes_sorted.loc[new_pairs].copy()
new_right_nodes.index.name = "skid_right"

pair_df = pd.concat(
    (
        new_left_nodes.index.to_series().reset_index(drop=True),
        new_right_nodes.index.to_series().reset_index(drop=True),
        new_left_nodes["p_matched"].reset_index(drop=True),
    ),
    axis=1,
)
# remove anything that never actually got matched (not equal sized graphs)
pair_df = pair_df[pair_df["p_matched"] > 0].copy()
pair_df = pair_df.sort_values("p_matched", ascending=False).reset_index(drop=True)
pair_df.to_csv(OUT_PATH / "pair_predictions.csv")
pair_df
#%%
# plot the distribution of matching probabilities
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(data=pair_df, x="p_matched")

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


volume_names = ["cns"]
import navis
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_paired_neurons(left_ids, right_ids, dist=3):
    n_show = len(left_ids)
    n_cols = n_show + 1
    n_rows = 3
    # views = (dict(elev=-90, azim=90), dict(elev=0, azim=0), dict(elev=45, azim=45))
    views = (dict(elev=0, azim=-90), dict(elev=90, azim=-90), dict(elev=-45, azim=90))
    fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows), constrained_layout=True)
    gs = plt.GridSpec(n_rows, n_cols, figure=fig, hspace=0, wspace=0)
    colors = [subgraph_palette["LL"], subgraph_palette["RR"]]
    colors = [rgb2hex(*color) for color in colors]
    axs = np.empty((n_rows, n_cols), dtype="object")
    j = 0
    for i, view in enumerate(views):
        ax = fig.add_subplot(gs[(i, j)], projection="3d")
        axs[(i, j)] = ax
        volumes = [pymaid.get_volume(v) for v in volume_names]
        navis.plot2d(volumes, ax=ax, method="3d", autoscale=True)
        ax.azim = view["azim"]
        ax.elev = view["elev"]
        ax.dist = dist
        for c in ax.collections:
            if isinstance(c, Poly3DCollection):
                c.set_alpha(0.02)

    for j, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
        j += 1
        neurons = [int(left_id), int(right_id)]
        palette = dict(zip(neurons, colors))
        for i, view in enumerate(views):
            ax = fig.add_subplot(gs[(i, j)], projection="3d")
            axs[(i, j)] = ax
            simple_plot_neurons(
                neurons,
                palette=palette,
                ax=ax,
                force_bounds=False,
                autoscale=True,
                soma=False,
                dist=dist,
                lw=1.5,
                **view,
            )
            draw_box(ax, color="lightgrey")
    return fig, axs


#%%

import matplotlib.transforms as mtrans


def set_border(ax):
    ax.set(xticks=[], yticks=[])
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    # ax.spines[["left", "right", "top", "bottom"]].set_color("lightgrey")


def draw_axes_dividers(fig, axs, axis="x"):
    # Adapted from
    # REF: https://stackoverflow.com/questions/26084231/draw-a-separator-or-lines-between-subplots
    r = fig.canvas.get_renderer()

    def get_bbox(ax):
        return ax.get_tightbbox(r).transformed(fig.transFigure.inverted())

    bboxes = np.array(list(map(get_bbox, axs.flat)), mtrans.Bbox).reshape(axs.shape)

    if axis == "y":
        # Get the minimum and maximum extent, get the coordinate half-way between those
        ymax = (
            np.array(list(map(lambda b: b.y1, bboxes.flat)))
            .reshape(axs.shape)
            .max(axis=1)
        )
        ymin = (
            np.array(list(map(lambda b: b.y0, bboxes.flat)))
            .reshape(axs.shape)
            .min(axis=1)
        )
        ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

        # Draw a horizontal lines at those coordinates
        for y in ys:
            line = plt.Line2D(
                [0, 1], [y, y], transform=fig.transFigure, color="lightgrey"
            )
            fig.add_artist(line)
    elif axis == "x":
        xmax = (
            np.array(list(map(lambda b: b.x1, bboxes.flat)))
            .reshape(axs.shape)
            .max(axis=0)
        )
        xmin = (
            np.array(list(map(lambda b: b.x0, bboxes.flat)))
            .reshape(axs.shape)
            .min(axis=0)
        )
        xs = np.c_[xmax[1:], xmin[:-1]].mean(axis=1)

        # Draw a vertical lines at those coordinates
        for x in xs:
            line = plt.Line2D(
                [x, x], [0, 1], transform=fig.transFigure, color="lightgrey"
            )
            fig.add_artist(line)


def draw_compass_rose(labels, ax):
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    outs = [(0.5, 0.75), (0.5, 0.25), (0.25, 0.5), (0.75, 0.5)]
    h_alignments = ["center", "center", "right", "left"]
    v_alignments = ["bottom", "top", "center", "center"]
    for i, (out, label) in enumerate(zip(outs, labels)):
        ax.annotate(
            "",
            out,
            xytext=(0.5, 0.5),
            textcoords="data",
            arrowprops=dict(arrowstyle="-|>", linewidth=3),
        )
        ax.text(*out, label, ha=h_alignments[i], va=v_alignments[i])


def plot_neuron_pair_grid(left_ids, right_ids, title=""):
    n_show = len(left_ids)

    views = (("-x", "-y"), ("-x", "-z"), ("z", "-y"))
    n_views = len(views)
    n_col = n_show + 2
    size = 2
    fig, axs = plt.subplots(
        n_views,
        n_col,
        figsize=(size * n_col, size * n_views),
        # gridspec_kw=dict(hspace=0, wspace=0),
        # constrained_layout=True
    )
    volume = pymaid.get_volume("cns")

    colors = sns.color_palette("tab20")

    for i in range(n_views):
        ax = axs[i, 0]
        set_border(ax)

    for i, view in enumerate(views):
        ax = axs[i, 1]
        navis.plot2d(
            volume,
            method="2d",
            view=view,
            soma=True,
            volume_outlines=True,
            ax=ax,
        )
        ax.margins(0.05)
        ax.axis("equal")
        set_border(ax)

    for j, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
        neuron_ids = [left_id, right_id]
        neurons = pymaid.get_neurons(neuron_ids)
        for i, view in enumerate(views):
            ax = axs[i, j + 2]
            navis.plot2d(
                neurons,
                method="2d",
                view=view,
                colors=[colors[j * 2], colors[j * 2 + 1]],
                alpha=1,
                soma=True,
                ax=ax,
                autoscale=True,
                linewidth=1,
            )
            ax.margins(0.05)
            ax.axis("equal")
            set_border(ax)

            ax = axs[i, 1]
            navis.plot2d(
                neurons,
                method="2d",
                view=view,
                colors=[colors[j * 2], colors[j * 2 + 1]],
                alpha=1,
                soma=True,
                ax=ax,
                autoscale=True,
            )
            # ax.margins(0.02)
            # ax.axis("square")

            # ax.autoscale(enable=False)
            # navis.plot2d(
            #     volume,
            #     method="2d",
            #     view=view,
            #     soma=True,
            #     volume_outlines=True,
            #     ax=ax,
            #     autoscale=False,
            # )
            # ax.autoscale(enable=True)

    for ax in axs.flat:
        ax.margins(0.02)

    fig.tight_layout()
    draw_axes_dividers(fig, axs, "x")
    draw_axes_dividers(fig, axs, "y")

    axs[1, 0].set_ylabel("View", fontsize="x-large")

    draw_compass_rose(["D", "V", "L", "R"], axs[0, 0])
    draw_compass_rose(["A", "P", "L", "R"], axs[1, 0])
    draw_compass_rose(["D", "V", "A", "P"], axs[2, 0])

    ax = axs[2, 1]
    ax.annotate(
        "CNS boundary",
        (0.5, 0.3),
        xytext=(0, -40),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="center",
        arrowprops=dict(arrowstyle="->", linewidth=2),
    )

    axs[0, 1].set_title("All")
    for j in range(n_show):
        axs[0, j + 2].set_title(f"Match {j + 1}")

    fig.text(0.5, 1.02, title, fontsize="x-large", ha="center", va="bottom")
    fig.set_facecolor("w")
    return fig, ax


#%%
rng = np.random.default_rng(9)

# morphologies for some good matches
n_show = 6
glue("n_show", n_show)
best_pair_df = pair_df[pair_df["p_matched"] >= 0.999]
best_pair_df = best_pair_df.sample(n=n_show, replace=False, random_state=rng)

fig, ax = plot_neuron_pair_grid(
    best_pair_df["skid_left"],
    best_pair_df["skid_right"],
    title="Neuron matches (high confidence)",
)

gluefig("example_matched_morphologies_good", fig)
#%%
# morphologies for some bad matches
worst_pair_df = pair_df.sort_values("p_matched").iloc[:n_show]

fig, ax = plot_neuron_pair_grid(
    worst_pair_df["skid_left"],
    worst_pair_df["skid_right"],
    title="Neuron matches (low confidence)",
)

gluefig("example_matched_morphologies_bad", fig)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
