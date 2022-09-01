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
from pkg.plot import method_palette, set_theme
from tqdm.autonotebook import tqdm

DISPLAY_FIGS = True

FILENAME = "simulations_nonmatches"

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
n_sims = 1000
glue("n_sims", n_sims, form="long")
ipsi_rho = 0.8
glue("ipsi_rho", ipsi_rho)
ipsi_p = 0.3
glue("ipsi_p", ipsi_p)
contra_p = 0.2
glue("contra_p", contra_p)

#%% [markdown]
# - Let the directed correlated Erdos-Reyni model be written as $CorrER(n, p, \rho)$, where $n$ is
#   the number of nodes, $p$ is the density, and $\rho$ is the correlation between the
#   two networks.
# - The ipsilateral subgraphs were sampled from a $CorrER$ model:
#   - $A_{LL}^{'}, A_{RR}^{'} \sim CorrER(${glue:text}`simulations-n_side`, {glue:text}`simulations-ipsi_p`, {glue:text}`simulations-ipsi_rho`$)$
# - Independently from the ipsilateral networks, the contralateral subgraphs were also sampled from a $CorrER$ model:
#   - $A_{LR}^{'}, A_{RL}^{'} \sim CorrER(${glue:text}`simulations-n_side`, {glue:text}`simulations-contra_p`, $\rho_{contra})$
# - The full network was then defined as
#   - $A^{'} = \begin{bmatrix} A_{LL}^{'} & A_{LR}^{'}\\  A_{RL}^{'} & A_{RR}^{'} \end{bmatrix}$
# - A random permutation was applied to the nodes of the "right hemisphere" in each sampled network:
#   - $A = \begin{bmatrix} I_n & 0 \\ 0 & P_{rand} \end{bmatrix} A{'} \begin{bmatrix} I_n & 0 \\ 0 & P_{rand} \end{bmatrix}^T = \begin{bmatrix} A_{LL}^{'} & A_{LR}^{'} P_{rand}^T \\  P_{rand} A_{RL}^{'} & P_{rand} A_{RR}^{'} P_{rand}^T \end{bmatrix}$
#   - Thus we can write
#     - $A_{LL} = A_{LL}^{'}$
#     - $A_{RR} = P_{rand} A_{RR}^{'} P_{rand}^T$
#     - $A_{LR} = A_{LR}^{'} P_{rand}^T$
#     - $A_{RL} = P_{rand} A_{RL}^{'} $


#%% [markdown]
# ## Experiment
# - $\rho_{contra}$ was varied from 0 to 1.
# - For each value of $\rho_{contra}$, {glue:text}`simulations-n_sims` networks were sampled according to the model above.
# - For each sampled network, two algorithms were applied to try to recover the alignment between the left and the right:
#   - Graph matching **(GM)**, using only the ipsilateral subgraphs $A_{LR}$ and $A_{RR}$.
#     - $\min_{P} \|A_{LL} - P A_{RR} P^T\|_F^2$
#   - Bisected graph matching **(BGM)**, using ipsilateral and contralateral subgraphs:
#     - $\min_{P} \|A_{LL} - P A_{RR} P^T\|_F^2 + \|A_{LR} P^T - P A_{RL}\|_F^2$
# - Both algorithms were run with the same (default) settings: one initialization at the barycenter,
#   maximum 30 Frank-Wolfe (FW) iterations, stopping tolerance (on the norm of the difference
#   between solutions at each FW iteration) of 0.01.
# - For each sampled network and algorithm, we computed the matching accuracy for the
#   recovered permutation.

#%%

from graspologic.match import graph_match
from graspologic.simulations import er_np

rows = []

contra_rhos = [0.2, 0.4, 0.6, 0.8]
n_unmatched_range = np.arange(0, 11)
n_sims = 1000
pbar = tqdm(total=n_sims * len(n_unmatched_range) * len(contra_rhos), leave=False)
n_total = 20
for contra_rho in contra_rhos:
    for n_unmatched in n_unmatched_range:
        for sim in range(n_sims):

            # simulate the correlated subgraphs
            A_corr, B_corr = er_corr(n_side, ipsi_p, ipsi_rho, directed=True)
            AB_corr, BA_corr = er_corr(n_side, contra_p, contra_rho, directed=True)

            n_side = 
            # simulate the uncorrelated subgraphs
            n_total = n_side + n_unmatched
            A = er_np(n_total, ipsi_p, directed=True)
            B = er_np(n_total, ipsi_p, directed=True)
            AB = er_np(n_total, contra_p, directed=True)
            BA = er_np(n_total, contra_p, directed=True)

            A[:n_side, :n_side] = A_corr
            B[:n_side, :n_side] = B_corr
            AB[:n_side, :n_side] = AB_corr
            BA[:n_side, :n_side] = BA_corr

            # permute one side as appropriate
            perm = rng.permutation(n_total)
            undo_perm = np.argsort(perm)
            B = B[perm][:, perm]
            AB = AB[:, perm]
            BA = BA[perm, :]

            # run the matching
            for method in ["GM", "BGM"]:
                if method == "GM":
                    _, perm, _, _ = graph_match(A, B)
                elif method == "BGM":
                    _, perm, _, _ = graph_match(A, B, AB, BA)

                match_ratio = (perm == undo_perm)[:n_side].mean()

                rows.append(
                    {
                        "n_unmatched": n_unmatched,
                        "ipsi_rho": ipsi_rho,
                        "contra_rho": contra_rho,
                        "match_ratio": match_ratio,
                        "sim": sim,
                        "method": method,
                    }
                )
            pbar.update(1)

results = pd.DataFrame(rows)

#%%

results["p_unmatched"] = results["n_unmatched"] / n_side + 1

fig, axs = plt.subplots(1, len(contra_rhos), figsize=(20, 5), sharey=True)

for i, (contra_rho, group_data) in enumerate(results.groupby("contra_rho")):
    ax = axs.flat[i]
    sns.lineplot(data=group_data, x="p_unmatched", y="match_ratio", hue="method", ax=ax)

    ax.set_title(f"Contralateral correlation = {contra_rho}")
    if i == 0:
        sns.move_legend(ax, loc="upper right", frameon=True, title="Method")
    else:
        ax.get_legend().remove()

    ax.set(ylabel="Matching accuracy", xlabel="Added unmatched nodes")
    ax.tick_params(which="both", length=5)


#%% [markdown]
# ## End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
