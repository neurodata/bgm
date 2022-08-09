#%% [markdown]
# # Simulation
#%%

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.simulations import er_corr
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.match import GraphMatchSolver
from pkg.plot import method_palette, set_theme, dashes
from tqdm.autonotebook import tqdm

DISPLAY_FIGS = True

FILENAME = "size_simulations"

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
np.random.seed(88888888)


#%% [markdown]
# ## Model
#%%
n_side = 10
glue("n_side", n_side)
n_sims = 10
glue("n_sims", n_sims, form="long")
ipsi_rho = 0.8
glue("ipsi_rho", ipsi_rho)
ipsi_p = 0.3
glue("ipsi_p", ipsi_p)
contra_p = 0.2
glue("contra_p", contra_p)

#%%


# n_sides = (np.array([18, 22, 286, 360, 1240]) / 2).astype(int)
# sims_per_size = {9: 1000, 11: 1000, 143: 100, 180: 100, 620: 20}

rerun = False
n_sides = np.floor(np.geomspace(9, 620, 10)).astype(int)
sims_per_size = dict(zip(n_sides, np.floor((10000 / (n_sides))).astype(int)))
contra_rhos = [0.2, 0.4, 0.6]

if rerun:
    rows = []
    pbar = tqdm(total=len(contra_rhos) * sum([sims_per_size[size] for size in n_sides]))
    for contra_rho in contra_rhos:
        for n_side in n_sides:
            for sim in range(sims_per_size[n_side]):
                # simulate the correlated subgraphs
                A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
                AB, BA = er_corr(n_side, contra_p, contra_rho, directed=True)

                # permute one side as appropriate
                perm = rng.permutation(n_side)
                undo_perm = np.argsort(perm)
                B = B[perm][:, perm]
                AB = AB[:, perm]
                BA = BA[perm, :]

                # run the matching
                for method in ["GM", "BGM"]:
                    if method == "GM":
                        solver = GraphMatchSolver(A, B)
                    elif method == "BGM":
                        solver = GraphMatchSolver(A, B, AB=AB, BA=BA)
                    solver.solve()
                    match_ratio = (solver.permutation_ == undo_perm).mean()

                    rows.append(
                        {
                            "ipsi_rho": ipsi_rho,
                            "contra_rho": contra_rho,
                            "match_ratio": match_ratio,
                            "sim": sim,
                            "method": method,
                            "n_side": n_side,
                        }
                    )
                pbar.update(1)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH / "matching_results.csv")
else:
    results = pd.read_csv(OUT_PATH / "matching_results.csv", index_col=0)

#%%


fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)

for i, (contra_rho, group_data) in enumerate(results.groupby("contra_rho")):
    ax = axs.flat[i]
    sns.lineplot(
        data=group_data,
        x="n_side",
        y="match_ratio",
        hue="method",
        style='method',
        ax=ax,
        palette=method_palette,
        dashes=dashes,
    )
    ax.set_title(f"Contralateral correlation = {contra_rho}")
    if i == 0:
        sns.move_legend(ax, loc="upper right", frameon=True, title="Method")
    else:
        ax.get_legend().remove()
    ax.set(ylabel="Matching accuracy", xlabel="Nodes per side")
    ax.tick_params(which="both", length=5)

gluefig("matching_accuracy_size", fig)

