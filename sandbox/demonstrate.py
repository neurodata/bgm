#%% 
#%%

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from giskard.plot import scattermap
from graspologic.simulations import er_corr
from matplotlib.patheffects import Normal, Stroke
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.match import BisectedGraphMatchSolver, GraphMatchSolver
from pkg.plot import set_theme, subgraph_palette


DISPLAY_FIGS = True

FILENAME = "demonstrate"

OUT_PATH = OUT_PATH / FILENAME

FIG_PATH = FIG_PATH / FILENAME


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

#%%
n_side = 10
n_sims = 1000
ipsi_rho = 0.8
p = 0.3

#%%
contra_rho = 0.8
A, B = er_corr(n_side, p, ipsi_rho, directed=True)
AB, BA = er_corr(n_side, p, contra_rho, directed=True)

# construct the full network
indices_A = np.arange(n_side)
indices_B = np.arange(n_side, 2 * n_side)
adjacency = np.zeros((2 * n_side, 2 * n_side))
adjacency[np.ix_(indices_A, indices_A)] = A
adjacency[np.ix_(indices_B, indices_B)] = B
adjacency[np.ix_(indices_A, indices_B)] = AB
adjacency[np.ix_(indices_B, indices_A)] = BA

# permute one hemisphere
side_perm = np.random.permutation(n_side) + n_side
perm = np.concatenate((indices_A, side_perm))
adjacency = adjacency[np.ix_(perm, perm)]
undo_perm = np.argsort(side_perm)

#%%

left_inds = np.arange(len(A))
right_inds = np.arange(len(B)) + len(A)
row_inds, col_inds = np.nonzero(adjacency)

source_is_left = np.isin(row_inds, left_inds)
target_is_left = np.isin(col_inds, left_inds)

colors = np.full(len(row_inds), "xx")
colors[source_is_left & target_is_left] = "LL"
colors[~source_is_left & ~target_is_left] = "RR"
colors[source_is_left & ~target_is_left] = "LR"
colors[~source_is_left & target_is_left] = "RL"

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

scattermap(
    adjacency,
    hue=colors,
    palette=subgraph_palette,
    sizes=(450, 450),
    marker="s",
    ax=ax,
)
ax.spines.top.set_visible(True)
ax.spines.right.set_visible(True)
fig.set_facecolor("w")

n_right = len(right_inds)
n_left = len(left_inds)
div_kws = dict(color="grey", linewidth=3, linestyle=":")
ax.axvline(n_left - 0.5, **div_kws)
ax.axhline(n_left - 0.5, **div_kws)

ticks = [n_left / 2 - 0.5, n_left + n_right / 2 - 0.5]
ax.set(xticks=ticks, yticks=ticks)
ax.xaxis.tick_top()
texts = ax.set_xticklabels(["Left", "Right"], fontsize="xx-large")
texts = ax.set_yticklabels(["Left", "Right"], fontsize="xx-large")


ax.tick_params(axis="both", which="major", pad=10)


def get_pos(s):
    if s == "L":
        return n_left / 2 - 0.5
    elif s == "R":
        return n_left / 2 + n_right - 0.5


def nice_text(source, target, edgecolor="black"):
    s = source + r"$\rightarrow$" + target
    x = get_pos(target)
    y = get_pos(source)
    color = subgraph_palette[f"{source}{target}"]
    text = ax.text(
        x,
        y,
        s,
        color=color,
        fontsize="xx-large",
        transform=ax.transData,
        ha="center",
        va="center",
        fontweight="bold",
    )
    text.set_path_effects([Stroke(linewidth=4, foreground=edgecolor), Normal()])


nice_text("L", "L")
nice_text("R", "R")
nice_text("L", "R")
nice_text("R", "L")


ax.spines[:].set_color("grey")

gluefig("adjacencies", fig)

#%%

perm = rng.permutation(10)


def draw_permutation(permutation, y=0, bump=0):
    # ax.axhline(0.5, color="black")
    # ax.set_xlim((-0.5, 9.5))

    for target, source in enumerate(permutation):
        if target > source:
            sign = "-"
        else:
            sign = ""
        ax.annotate(
            "",
            xy=(target + bump, y),
            xytext=(source + bump, y),
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={sign}0.5",
                facecolor="black",
            ),
        )


def draw_label(source, target, edgecolor="black"):
    s = source + r"$\rightarrow$" + target
    x, y = n_side / 2 - 0.5
    color = subgraph_palette[f"{source}{target}"]
    text = ax.text(
        x,
        y,
        s,
        color=color,
        fontsize="xx-large",
        transform=ax.transData,
        ha="center",
        va="center",
        fontweight="bold",
    )
    text.set_path_effects([Stroke(linewidth=4, foreground=edgecolor), Normal()])


fig, ax = plt.subplots(3, 3, gridspec_kw=dict(width_ratios=[1, 1, 1]))


# %%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
