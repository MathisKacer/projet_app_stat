# %% [markdown]
# # 02 — Prétraitement
#
# Chaque étape appelle une fonction de `data/preprocess.py` et affiche
# quelques statistiques de contrôle (pondérées par `pm17B` sauf mention).
#
# 1. Cible `Q19A` : suppression des âges > 25 ans, discrétisation par quantiles,
#    NA → « Non concerné » (jamais bu).
# 2. Situation : `Q04` complété, « Non concerné » pour les questions filtrées,
#    variable de synthèse `SITUATION`.
# 3. Autres variables : NA → « Non répondu ».

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd() if (Path.cwd() / "data").is_dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from data import load_escap_data  # noqa: E402
from data.labels import NON_CONCERNE, NON_REPONDU, PREDICTORS, TARGET, WEIGHT  # noqa: E402
from data.preprocess import (  # noqa: E402
    AGE_MAX,
    SITUATION,
    SITUATION_VARS,
    TARGET_CLASS,
    build_situation,
    clean_target,
    complete_q04,
    discretize_target,
    encode_missing,
    target_thresholds,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8985"
TEXT = "#52514e"
BLUE_RAMP = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]  # classes d'âge, clair -> foncé

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.labelcolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)


def weighted_share(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Part pondérée (%) de chaque modalité."""
    s = weights.groupby(values, observed=False).sum()
    return s / s.sum() * 100


def summary(values: pd.Series, weights: pd.Series) -> pd.DataFrame:
    """Effectif et part pondérée de chaque modalité."""
    return pd.DataFrame(
        {"effectif": values.value_counts(sort=False), "% pondéré": weighted_share(values, weights)}
    ).round(1)


# %%
raw = load_escap_data()
print(f"Données brutes : {raw.shape[0]} individus")

# %% [markdown]
# ## 1. Cible `Q19A`
#
# ### 1.a Suppression des âges > 25 ans
#
# La JDC se fait au plus tard à 25 ans : un âge au premier alcool supérieur
# est impossible.

# %%
df1 = clean_target(raw)
removed = raw.loc[~raw.index.isin(df1.index), TARGET]
print(f"{len(removed)} individus supprimés (Q19A > {AGE_MAX}) : {removed.tolist()}")
print(f"Reste : {len(df1)} individus")

# %% [markdown]
# ### 1.b Distribution de `Q19A` : histogramme et boîte à moustaches
#
# Statistiques pondérées, calculées sur les individus ayant déjà bu. Les
# barres sont colorées selon la classe de quantile à laquelle elles
# appartiennent (voir 1.c).


# %%
def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, cw = np.asarray(values)[order], np.cumsum(np.asarray(weights)[order])
    return v[np.searchsorted(cw, np.asarray(q) * cw[-1])]


def weighted_box_stats(values, weights):
    """Statistiques d'une boîte à moustaches pondérée (format `Axes.bxp`)."""
    values, weights = np.asarray(values), np.asarray(weights)
    q1, med, q3 = weighted_quantile(values, weights, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
    return {
        "med": med,
        "q1": q1,
        "q3": q3,
        "whislo": inside.min(),
        "whishi": inside.max(),
        "mean": np.average(values, weights=weights),
        "fliers": np.unique(values[(values < inside.min()) | (values > inside.max())]),
    }


obs = df1[TARGET].notna()
ages, w_ages = df1.loc[obs, TARGET], df1.loc[obs, WEIGHT]
thresholds = target_thresholds(df1)
box = weighted_box_stats(ages, w_ages)

dist = weighted_share(ages, w_ages)
class_idx = np.searchsorted(thresholds, dist.index, side="left")

fig, (ax_box, ax_hist) = plt.subplots(
    2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [1, 4]}
)
ax_box.bxp(
    [box],
    orientation="horizontal",
    showmeans=True,
    widths=0.6,
    patch_artist=True,
    boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE},
    medianprops={"color": BLUE, "linewidth": 2},
    whiskerprops={"color": BLUE},
    capprops={"color": BLUE},
    meanprops={"marker": "D", "markerfacecolor": ORANGE, "markeredgecolor": "white"},
    flierprops={"marker": "o", "markersize": 4, "markerfacecolor": GREY, "markeredgecolor": GREY},
)
ax_box.set_yticks([])
ax_box.spines["left"].set_visible(False)
ax_box.grid(False)
ax_box.set_title("Âge au premier alcool (déjà buveurs, pondéré)")

ax_hist.bar(dist.index, dist.values, color=[BLUE_RAMP[i] for i in class_idx], width=0.8)
for t in thresholds:
    ax_hist.axvline(t + 0.5, color=TEXT, linestyle=":", linewidth=1)
ax_hist.set_xticks(dist.index.astype(int))
ax_hist.set_xlabel("Âge au premier alcool (ans)")
ax_hist.set_ylabel("Part (%, pondéré)")
ax_hist.grid(axis="x", visible=False)
plt.tight_layout()
plt.show()

pd.Series(
    {
        "Effectif (déjà bu)": len(ages),
        "Moyenne": box["mean"],
        "Écart-type": np.sqrt(np.average((ages - box["mean"]) ** 2, weights=w_ages)),
        "Min": ages.min(),
        "Q1": box["q1"],
        "Médiane": box["med"],
        "Q3": box["q3"],
        "Max": ages.max(),
        "Moustaches": f"[{box['whislo']:.0f}, {box['whishi']:.0f}]",
        "Valeurs hors moustaches": f"{(~ages.between(box['whislo'], box['whishi'])).sum()} ind.",
    }
).to_frame("Q19A")

# %% [markdown]
# ### 1.c Discrétisation par quantiles
#
# L'âge étant entier et très concentré, les quartiles tombent sur des
# valeurs répétées : pour chaque quartile, on retient l'âge dont la
# fonction de répartition pondérée est la plus proche, ce qui donne des
# classes de taille proche. Les NA deviennent « Non concerné » (jamais bu).

# %%
df1 = discretize_target(df1)
print(f"Seuils retenus : {thresholds}")

stats_class = summary(df1[TARGET_CLASS], df1[WEIGHT])
stats_class["% pondéré parmi buveurs"] = (
    weighted_share(df1.loc[obs, TARGET_CLASS], df1.loc[obs, WEIGHT])
    .drop(NON_CONCERNE)
    .round(1)
)
stats_class["âge moyen"] = (
    df1[obs]
    .groupby(TARGET_CLASS, observed=False)[[TARGET, WEIGHT]]
    .apply(lambda g: np.average(g[TARGET], weights=g[WEIGHT]) if len(g) else np.nan)
    .round(2)
)
stats_class

# %%
shares = weighted_share(df1[TARGET_CLASS], df1[WEIGHT])
fig, ax = plt.subplots(figsize=(8, 3.2))
bars = ax.barh(shares.index.astype(str), shares.values, color=BLUE_RAMP + [GREY], height=0.7)
ax.bar_label(bars, fmt="%.1f %%", padding=3, fontsize=8, color=TEXT)
ax.invert_yaxis()
ax.set_xlim(0, shares.max() * 1.2)
ax.set_xlabel("Part (%, pondéré)")
ax.set_title("Classes d'âge au premier alcool (Q19A_CLASSE)")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Situation (`Q04`, `Q04A`, `Q04B`)
#
# ### 2.a Complétion de `Q04`
#
# `Q04A` n'est posée qu'à ceux qui suivent des études (`Q04` = 1) et `Q04B`
# qu'à ceux qui les ont arrêtées (`Q04` = 2). Si `Q04` manque et qu'une seule
# des deux est renseignée, on en déduit `Q04`. Si les deux sont renseignées,
# on ne peut pas trancher et `Q04` reste non répondu.

# %%
patterns = (
    df1[list(SITUATION_VARS)]
    .notna()
    .value_counts()
    .rename("effectif")
    .reset_index()
    .replace({True: "renseigné", False: "NA"})
)
patterns

# %%
completed = complete_q04(df1)
filled = df1["Q04"].isna() & completed["Q04"].notna()
print(f"Q04 manquant avant : {df1['Q04'].isna().sum()}")
print(f"  complété en « Études » (1)          : {(filled & (completed['Q04'] == 1)).sum()}")
print(f"  complété en « Études arrêtées » (2) : {(filled & (completed['Q04'] == 2)).sum()}")
print(f"Q04 manquant après : {completed['Q04'].isna().sum()}")

incoherent = ((completed["Q04"] == 1) & completed["Q04B"].notna()) | (
    (completed["Q04"] == 2) & completed["Q04A"].notna()
)
print(f"Réponses hors filtre (Q04 fait foi, recodées « Non concerné ») : {incoherent.sum()}")

# %% [markdown]
# ### 2.b Recodage et variable `SITUATION`
#
# - « Non concerné » : la question ne devait pas être posée compte tenu de `Q04`.
# - « Non répondu » : la question devait être posée mais n'a pas de réponse.

# %%
df2 = build_situation(df1)
pd.crosstab(df2["Q04"], df2["Q04A"], margins=True, margins_name="Total")

# %%
pd.crosstab(df2["Q04"], df2["Q04B"], margins=True, margins_name="Total")

# %%
stats_sit = summary(df2[SITUATION], df2[WEIGHT])
stats_sit["% jamais bu"] = (
    df2.assign(nb=df2[TARGET_CLASS].eq(NON_CONCERNE) * df2[WEIGHT])
    .groupby(SITUATION, observed=False)[["nb", WEIGHT]]
    .sum()
    .pipe(lambda g: g["nb"] / g[WEIGHT] * 100)
    .round(1)
)
drinkers = df2[TARGET].notna()
stats_sit["âge moyen 1er alcool"] = (
    df2[drinkers]
    .groupby(SITUATION, observed=False)[[TARGET, WEIGHT]]
    .apply(lambda g: np.average(g[TARGET], weights=g[WEIGHT]) if len(g) else np.nan)
    .round(2)
)
stats_sit

# %%
fig, ax = plt.subplots(figsize=(8, 4))
colors = [GREY if NON_REPONDU in m else (BLUE if m.startswith("Études") else ORANGE)
          for m in stats_sit.index]
bars = ax.barh(stats_sit.index.astype(str), stats_sit["% pondéré"], color=colors, height=0.7)
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8, color=TEXT)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Part (%, pondéré, échelle log)")
ax.set_title("Variable SITUATION")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Autres variables : NA → « Non répondu »

# %%
others = [v for v in PREDICTORS if v not in SITUATION_VARS]
df3 = encode_missing(df2, others)

na_table = pd.DataFrame(
    {
        "Non répondu (effectif)": df2[others].isna().sum(),
        "Non répondu (% pondéré)": [
            np.average(df2[v].isna(), weights=df2[WEIGHT]) * 100 for v in others
        ],
    }
).round(1)
na_table.sort_values("Non répondu (effectif)", ascending=False)

# %%
print("Modalités après recodage :")
for var in others:
    counts = df3[var].value_counts(sort=False)
    print(f"  {var:6s} " + " | ".join(f"{m}: {n}" for m, n in counts.items()))

# %% [markdown]
# ## Bilan

# %%
print(f"Jeu prétraité : {df3.shape[0]} individus, {df3.shape[1]} colonnes")
remaining_na = df3.isna().sum()
print("NA restants :", remaining_na[remaining_na > 0].to_dict(), "(Q19A continue des non-buveurs)")
df3.head()
