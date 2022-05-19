import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def matched_stripplot(
    data,
    x=None,
    y=None,
    jitter=0.2,
    hue=None,
    match=None,
    ax=None,
    matchline_kws=None,
    order=None,
    **kwargs,
):
    data = data.copy()
    if ax is None:
        ax = plt.gca()

    if order is None:
        unique_x_var = data[x].unique()
    else:
        unique_x_var = order
    ind_map = dict(zip(unique_x_var, range(len(unique_x_var))))
    data["x"] = data[x].map(ind_map)
    if match is not None:
        groups = data.groupby(match)
        for _, group in groups:
            perturb = np.random.uniform(-jitter, jitter)
            data.loc[group.index, "x"] += perturb
    else:
        data["x"] += np.random.uniform(-jitter, jitter, len(data))

    sns.scatterplot(data=data, x="x", y=y, hue=hue, ax=ax, zorder=1, **kwargs)

    if match is not None:
        unique_match_var = data[match].unique()
        fake_palette = dict(zip(unique_match_var, len(unique_match_var) * ["black"]))
        if matchline_kws is None:
            matchline_kws = dict(alpha=0.2, linewidth=1)
        sns.lineplot(
            data=data,
            x="x",
            y=y,
            hue=match,
            ax=ax,
            legend=False,
            palette=fake_palette,
            zorder=-1,
            **matchline_kws,
        )
    ax.set(xlabel=x, xticks=np.arange(len(unique_x_var)), xticklabels=unique_x_var)
    return ax
