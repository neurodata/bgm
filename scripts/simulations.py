#%%
# Simulation

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
from pkg.match import BisectedGraphMatchSolver, GraphMatchSolver
from pkg.plot import method_palette, set_theme
from tqdm import tqdm

DISPLAY_FIGS = True

FILENAME = "simulations"

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
glue("n_side", n_side)
n_sims = 1000
glue("n_sims", n_sims, form="long")
ipsi_rho = 0.8
glue("ipsi_rho", ipsi_rho)
ipsi_p = 0.3
glue("ipsi_p", ipsi_p)
contra_p = 0.2
glue("contra_p", contra_p)

rows = []
for contra_rho in np.linspace(0, 1, 11):
    for sim in tqdm(range(n_sims)):
        # simulate the correlated subgraphs
        A, B = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
        AB, BA = er_corr(n_side, contra_p, contra_rho, directed=True)

        # construct the full network
        indices_A = np.arange(n_side)
        indices_B = np.arange(n_side, 2 * n_side)
        adjacency = np.zeros((2 * n_side, 2 * n_side))
        adjacency[np.ix_(indices_A, indices_A)] = A
        adjacency[np.ix_(indices_B, indices_B)] = B
        adjacency[np.ix_(indices_A, indices_B)] = AB
        adjacency[np.ix_(indices_B, indices_A)] = BA

        # permute one hemisphere
        side_perm = rng.permutation(n_side) + n_side
        perm = np.concatenate((indices_A, side_perm))
        adjacency = adjacency[np.ix_(perm, perm)]
        undo_perm = np.argsort(side_perm)

        # run the matching
        for Solver, method in zip(
            [BisectedGraphMatchSolver, GraphMatchSolver], ["BGM", "GM"]
        ):
            solver = Solver(adjacency, indices_A, indices_B)
            solver.solve()
            match_ratio = (solver.permutation_ == undo_perm).mean()

            rows.append(
                {
                    "ipsi_rho": ipsi_rho,
                    "contra_rho": contra_rho,
                    "match_ratio": match_ratio,
                    "sim": sim,
                    "method": method,
                }
            )

results = pd.DataFrame(rows)

#%%
zero_acc = results[results["contra_rho"] == 0].groupby("method")["match_ratio"].mean()
zero_diff = zero_acc[1] - zero_acc[0]
glue("zero_diff", zero_diff, form="2.0f%")

point_9_acc = (
    results[results["contra_rho"] == 0.9].groupby("method")["match_ratio"].mean()
)
point_9_diff = point_9_acc[0] - point_9_acc[1]
glue("point_9_diff", point_9_diff, form="2.0f%")
#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(
    data=results,
    x="contra_rho",
    y="match_ratio",
    hue="method",
    style="method",
    hue_order=["GM", "BGM"],
    # dashes={"GM": "--", "BGM": "-"},
    ax=ax,
    palette=method_palette,
)
ax.set_ylabel("Matching accuracy")
ax.set_xlabel("Contralateral edge correlation")
sns.move_legend(ax, loc="upper left", title="Method", frameon=True)
gluefig("match_ratio_by_contra_rho", fig)

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
