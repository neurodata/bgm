#%%
import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.embed import AdjacencySpectralEmbed
from pkg.data import load_split_connectome
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import get_hemisphere_indices


FILENAME = "rdpg_sweep"

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
#%%

dataset = "maggot_subset"
adj, nodes = load_split_connectome(dataset, weights=False)

left_inds, right_inds = get_hemisphere_indices(nodes)
left_adj = adj[left_inds][:, left_inds]
right_adj = adj[right_inds][:, right_inds]
# AB = adj[left_inds][:, right_inds]
# BA = adj[right_inds][:, left_inds]

#%%
max_rank = 64
ase = AdjacencySpectralEmbed(n_components=max_rank)
left_X, left_Y = ase.fit_transform(left_adj)
right_X, right_Y = ase.fit_transform(right_adj)


#%%
def make_P(X, Y, rank, pad):
    P = X[:, :rank] @ Y[:, :rank].T
    P[P < pad] = pad
    P[P > 1 - pad] = 1 - pad
    return P


def compute_log_likelihood(adj, P):
    triu_inds = np.triu_indices_from(adj, k=1)
    tril_inds = np.tril_indices_from(adj, k=-1)
    flat_adj = np.concatenate((adj[triu_inds], adj[tril_inds]))
    flat_P = np.concatenate((P[triu_inds], P[tril_inds]))
    likelihood = flat_P**flat_adj * (1 - flat_P) ** (1 - flat_adj)
    log_likelihood = np.sum(np.log(likelihood))
    return log_likelihood


pad = 1 / (len(left_adj) ** 3)

rows = []
for rank in np.arange(1, max_rank):
    for source in ["left", "right"]:
        if source == "left":
            P = make_P(left_X, left_Y, rank, pad)
        else:
            P = make_P(right_X, right_Y, rank, pad)
        for target in ["left", "right"]:
            if target == "left":
                adj = left_adj
            else:
                adj = right_adj
            log_lik = compute_log_likelihood(adj, P)
            rows.append(
                {"rank": rank, "log_lik": log_lik, "source": source, "target": target}
            )

results = pd.DataFrame(rows)

#%%
set_theme()
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
train = results[results["source"] == results["target"]]
test = results[results["source"] != results["target"]]

ax = axs[0]
sns.lineplot(data=train, x="rank", y="log_lik", hue="target", ax=ax)
ax.set(ylabel="Log likelihood", xlabel="Rank")

ax = axs[1]
sns.lineplot(data=test, x="rank", y="log_lik", hue="target", ax=ax)
ax.set(xlabel="Rank")

idxmaxs = test.groupby("target")["log_lik"].idxmax()
maxs = test.loc[idxmaxs]

x = maxs.iloc[0]["rank"]
y = maxs.iloc[0]["log_lik"]
plt.autoscale(False)
ax.plot((x, x), (ax.get_ylim()[0], y))

maxs

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
