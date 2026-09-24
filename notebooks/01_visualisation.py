# %% [markdown]
# # 01 — Visualisation des données ESCAPAD 2017
#
# Premier tour d'horizon du jeu de données avant la modélisation de `Q19A`
# (âge au premier alcool) :
#
# 1. structure et valeurs manquantes ;
# 2. distribution de la cible et valeurs aberrantes ;
# 3. poids de sondage `pm17B` ;
# 4. distribution des prédicteurs ;
# 5. âge au premier alcool selon chaque prédicteur ;
# 6. qui ne renseigne pas `Q19A` ?
# 7. liens entre prédicteurs (V de Cramér).
#
# Toutes les statistiques sont **pondérées par `pm17B`** sauf mention contraire.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Permet d'importer le package `data` depuis le dossier notebooks/
ROOT = Path.cwd() if (Path.cwd() / "data").is_dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from data import load_escap_data  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# Palette : une couleur principale, une secondaire, des gris pour le reste.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8985"
TEXT = "#52514e"
# Rampe séquentielle bleue (clair -> foncé) pour les variables ordinales.
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

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

# %% [markdown]
# ## Chargement et dictionnaire des variables

# %%
df = load_escap_data()

TARGET = "Q19A"
WEIGHT = "pm17B"

# Libellés des variables et des modalités (d'après le sujet).
VARIABLES = {
    "Q03": ("Sexe", {1: "Homme", 2: "Femme"}),
    "Q04": ("Situation", {1: "Études", 2: "Études arrêtées"}),
    "Q04A": (
        "Situation scolaire",
        {1: "Lycée / collège", 2: "Apprentissage", 3: "Supérieur"},
    ),
    "Q04B": (
        "Situation professionnelle",
        {1: "Sans activité", 2: "Recherche emploi", 3: "Insertion", 4: "Travaille"},
    ),
    "Q05": ("Redoublement", {1: "Non", 2: "Oui"}),
    "Q06A": ("Difficultés pour lire", {1: "Non", 2: "Parfois", 3: "Souvent"}),
    "Q06B": ("Difficultés pour écrire", {1: "Non", 2: "Parfois", 3: "Souvent"}),
    "Q08": (
        "Lieu de vie",
        {
            1: "Chez parent(s)",
            2: "Internat",
            3: "Foyer / fam. accueil",
            4: "Propre logement",
            5: "Autre",
        },
    ),
    "Q08C": (
        "Vie des parents",
        {
            1: "Ensemble",
            2: "Séparés",
            3: "Jamais connus",
            4: "Décès",
            5: "Autre",
        },
    ),
    "Q09A1": (
        "Situation du père",
        {
            1: "Travaille",
            2: "Chômeur",
            3: "Au foyer",
            4: "Invalidité",
            5: "Retraité",
            6: "Ne sait pas",
            7: "Non concerné",
        },
    ),
    "Q09B1": (
        "Situation de la mère",
        {
            1: "Travaille",
            2: "Chômeuse",
            3: "Au foyer",
            4: "Invalidité",
            5: "Retraitée",
            6: "Ne sait pas",
            7: "Non concernée",
        },
    ),
    "Q10A1": ("PCS du père", None),
    "Q10B1": ("PCS de la mère", None),
    "B08A": ("Père boit à la maison", None),
    "B08B": ("Mère boit à la maison", None),
}
PCS = {
    1: "Agriculteur",
    2: "Artisan, commerçant",
    3: "Chef entr. ≥ 10 sal.",
    4: "Cadre, prof. sup.",
    5: "Prof. intermédiaire",
    6: "Employé",
    7: "Ouvrier",
    8: "Sans profession",
    9: "Non concerné",
}
ALCOOL_PARENT = {
    1: "Jamais",
    2: "Peu / an",
    3: "1-2 / mois",
    4: "≥ 1 / semaine",
    5: "Presque chaque jour",
}
for var in ("Q10A1", "Q10B1"):
    VARIABLES[var] = (VARIABLES[var][0], PCS)
for var in ("B08A", "B08B"):
    VARIABLES[var] = (VARIABLES[var][0], ALCOOL_PARENT)

PREDICTORS = list(VARIABLES)
MISSING_LABEL = "Non renseigné"


def as_labels(var: str) -> pd.Series:
    """Variable catégorielle avec libellés, les NaN devenant une modalité."""
    _, labels = VARIABLES[var]
    order = list(labels.values()) + [MISSING_LABEL]
    values = df[var].map(labels).fillna(MISSING_LABEL)
    return pd.Categorical(values, categories=order, ordered=True)


df.head()

# %%
print(f"{df.shape[0]} individus, {df.shape[1]} colonnes")
print(f"Identifiant unique : {df['A01'].is_unique}")
df.describe().T.round(2)

# %% [markdown]
# ## 1. Valeurs manquantes
#
# `Q04A` et `Q04B` sont des questions filtrées (selon `Q04`), leurs manquants
# sont donc en grande partie structurels.

# %%
na_rate = df.drop(columns=["A01", WEIGHT]).isna().mean().sort_values() * 100

fig, ax = plt.subplots(figsize=(7, 5))
colors = [ORANGE if v == TARGET else BLUE for v in na_rate.index]
bars = ax.barh(
    [f"{v} · {VARIABLES[v][0]}" if v in VARIABLES else f"{v} · Âge 1er alcool (cible)" for v in na_rate.index],
    na_rate.values,
    color=colors,
    height=0.7,
)
ax.bar_label(bars, fmt="%.1f %%", padding=3, fontsize=7, color=TEXT)
ax.set_xlabel("Part de valeurs manquantes (%, non pondéré)")
ax.set_title("Valeurs manquantes par variable")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, 105)
plt.tight_layout()
plt.show()

# %% [markdown]
# Cohérence du filtre `Q04` → `Q04A` / `Q04B` : on regarde les combinaisons
# de présence des trois variables.

# %%
filtre = df[["Q04", "Q04A", "Q04B"]].notna().value_counts().rename("effectif")
filtre.to_frame()

# %%
pd.crosstab(df["Q04"].fillna(0), df["Q04A"].fillna(0), margins=True).rename(
    index={0: "NA"}, columns={0: "NA"}
)

# %% [markdown]
# ## 2. La cible : âge au premier alcool (`Q19A`)
#
# Les appelés ont environ 17 ans : un âge au premier alcool supérieur à 18 ans
# est incohérent, et les très jeunes âges (≤ 5 ans) sont douteux. On les
# signale ici sans les supprimer.

# %%
AGE_MIN, AGE_MAX = 6, 18  # plage jugée plausible (à discuter)

y = df[TARGET]
w = df[WEIGHT]
obs = y.notna()

dist = (
    pd.DataFrame({"age": y[obs], "w": w[obs]})
    .groupby("age")["w"]
    .sum()
    .pipe(lambda s: s / s.sum() * 100)
)
plausible = (dist.index >= AGE_MIN) & (dist.index <= AGE_MAX)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(dist.index[plausible], dist[plausible], color=BLUE, width=0.8, label="Plage plausible")
ax.bar(dist.index[~plausible], dist[~plausible], color=ORANGE, width=0.8, label="Hors plage (suspect)")
wmean = np.average(y[obs], weights=w[obs])
ax.axvline(wmean, color=TEXT, linestyle="--", linewidth=1)
ax.annotate(
    f"moyenne pondérée\n{wmean:.2f} ans",
    xy=(wmean, dist.max() * 0.9),
    xytext=(wmean - 5.5, dist.max() * 0.85),
    fontsize=8,
    color=TEXT,
    arrowprops={"arrowstyle": "-", "color": GREY},
)
ax.set_xticks(dist.index.astype(int))
ax.set_xlabel("Âge au premier alcool (ans)")
ax.set_ylabel("Part des répondants (%, pondéré)")
ax.set_title("Distribution de l'âge au premier alcool")
ax.legend(loc="upper left")
ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.show()

hors_plage = obs & ((y < AGE_MIN) | (y > AGE_MAX))
print(f"Q19A renseigné : {obs.sum()} ({obs.mean():.1%})")
print(f"Hors plage [{AGE_MIN}, {AGE_MAX}] : {hors_plage.sum()} individus")
print(y[hors_plage].value_counts().sort_index().to_string())


# %%
def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, cw = np.asarray(values)[order], np.cumsum(np.asarray(weights)[order])
    return v[np.searchsorted(cw, q * cw[-1])]


y_obs, w_obs = y[obs], w[obs]
pd.Series(
    {
        "Moyenne non pondérée": y_obs.mean(),
        "Moyenne pondérée": wmean,
        "Écart-type pondéré": np.sqrt(np.average((y_obs - wmean) ** 2, weights=w_obs)),
        "Q1 pondéré": weighted_quantile(y_obs, w_obs, 0.25),
        "Médiane pondérée": weighted_quantile(y_obs, w_obs, 0.5),
        "Q3 pondéré": weighted_quantile(y_obs, w_obs, 0.75),
    }
).round(2).to_frame("Q19A")

# %% [markdown]
# ## 3. Poids de sondage `pm17B`

# %%
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(df[WEIGHT], bins=60, color=BLUE, edgecolor="white", linewidth=0.5)
ax.axvline(1, color=TEXT, linestyle="--", linewidth=1)
ax.set_xlabel("pm17B")
ax.set_ylabel("Effectif")
ax.set_title("Distribution des poids de sondage")
ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.show()

n_eff = w.sum() ** 2 / (w**2).sum()
print(f"Somme des poids : {w.sum():.0f} — taille effective (Kish) : {n_eff:.0f} / {len(w)}")

# %% [markdown]
# ## 4. Distribution des prédicteurs
#
# Parts pondérées de chaque modalité ; les non-réponses forment une modalité
# à part (en gris).


# %%
def weighted_shares(var: str) -> pd.Series:
    s = pd.DataFrame({"mod": as_labels(var), "w": w}).groupby("mod", observed=False)["w"].sum()
    return s / s.sum() * 100


ncols = 3
nrows = int(np.ceil(len(PREDICTORS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
for ax, var in zip(axes.flat, PREDICTORS):
    shares = weighted_shares(var)
    colors = [GREY if m == MISSING_LABEL else BLUE for m in shares.index]
    bars = ax.barh(shares.index.astype(str), shares.values, color=colors, height=0.7)
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7, color=TEXT)
    ax.invert_yaxis()
    ax.set_title(f"{var} · {VARIABLES[var][0]}")
    ax.set_xlim(0, shares.max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=7)
for ax in axes.flat[len(PREDICTORS):]:
    ax.remove()
fig.suptitle("Répartition des modalités (%, pondéré)", fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()

# %% [markdown]
# ## 5. Âge au premier alcool selon les prédicteurs
#
# Moyenne pondérée de `Q19A` par modalité, avec un intervalle de confiance à
# 95 % approché (taille effective de Kish). Seuls les âges dans la plage
# plausible sont retenus. La ligne pointillée est la moyenne globale.

# %%
clean = obs & (y >= AGE_MIN) & (y <= AGE_MAX)
mean_clean = np.average(y[clean], weights=w[clean])


def weighted_mean_ci(group: pd.DataFrame) -> pd.Series:
    yy, ww = group["y"], group["w"]
    if ww.sum() == 0:
        return pd.Series({"mean": np.nan, "ci": np.nan, "n": 0})
    m = np.average(yy, weights=ww)
    var = np.average((yy - m) ** 2, weights=ww)
    n_eff = ww.sum() ** 2 / (ww**2).sum()
    return pd.Series({"mean": m, "ci": 1.96 * np.sqrt(var / n_eff), "n": len(yy)})


def mean_by(var: str, mask=clean) -> pd.DataFrame:
    data = pd.DataFrame({"mod": as_labels(var), "y": y, "w": w})[mask]
    return data.groupby("mod", observed=False)[["y", "w"]].apply(weighted_mean_ci)


fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), sharex=True)
for ax, var in zip(axes.flat, PREDICTORS):
    stats = mean_by(var)
    stats = stats[stats["n"] >= 30]  # modalités trop rares écartées
    pos = np.arange(len(stats))
    colors = [GREY if m == MISSING_LABEL else BLUE for m in stats.index]
    ax.errorbar(
        stats["mean"], pos, xerr=stats["ci"], fmt="none", ecolor=colors, elinewidth=1.5, capsize=0
    )
    ax.scatter(stats["mean"], pos, color=colors, s=28, zorder=3, edgecolor="white", linewidth=1)
    ax.set_yticks(pos, [f"{m} (n={int(n)})" for m, n in zip(stats.index, stats["n"])])
    ax.invert_yaxis()
    ax.axvline(mean_clean, color=TEXT, linestyle=":", linewidth=1)
    ax.set_title(f"{var} · {VARIABLES[var][0]}")
    ax.grid(axis="y", visible=False)
for ax in axes.flat[len(PREDICTORS):]:
    ax.remove()
for ax in axes[-1]:
    ax.set_xlabel("Âge moyen au premier alcool (ans)")
fig.suptitle("Âge moyen au premier alcool par modalité (pondéré, IC 95 %)", fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()

# %% [markdown]
# ### Distribution selon le sexe

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for code, color in ((1, BLUE), (2, ORANGE)):
    m = clean & (df["Q03"] == code)
    d = pd.DataFrame({"age": y[m], "w": w[m]}).groupby("age")["w"].sum()
    d = d / d.sum() * 100
    ax.plot(d.index, d.values, color=color, linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", label=VARIABLES["Q03"][1][code])
ax.set_xticks(range(AGE_MIN, AGE_MAX + 1))
ax.set_xlabel("Âge au premier alcool (ans)")
ax.set_ylabel("Part (%, pondéré)")
ax.set_title("Distribution de l'âge au premier alcool selon le sexe")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Consommation d'alcool des deux parents
#
# Âge moyen au premier alcool pour chaque combinaison père × mère (cases
# de moins de 30 répondants masquées).

# %%
m = clean & df["B08A"].notna() & df["B08B"].notna()
grid = pd.DataFrame({"pere": df["B08A"], "mere": df["B08B"], "y": y, "w": w})[m]
grid["yw"] = grid["y"] * grid["w"]
agg = grid.groupby(["pere", "mere"]).agg(yw=("yw", "sum"), w=("w", "sum"), n=("y", "size"))
heat = (agg["yw"] / agg["w"]).where(agg["n"] >= 30).unstack()
counts = agg["n"].unstack()

fig, ax = plt.subplots(figsize=(7, 5.5))
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP[::-1])  # foncé = précoce
im = ax.imshow(heat.values, cmap=cmap, aspect="auto")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}\n(n={counts.values[i, j]})", ha="center", va="center",
                    fontsize=7, color="white" if v < heat.stack().median() else "#0b0b0b")
labels = list(ALCOOL_PARENT.values())
ax.set_xticks(range(5), labels, rotation=30, ha="right")
ax.set_yticks(range(5), labels)
ax.set_xlabel("La mère boit à la maison (B08B)")
ax.set_ylabel("Le père boit à la maison (B08A)")
ax.set_title("Âge moyen au premier alcool selon la consommation des parents")
ax.grid(False)
fig.colorbar(im, ax=ax, label="Âge moyen (ans)", shrink=0.8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Qui ne renseigne pas l'âge au premier alcool ?
#
# Les 18 % de `Q19A` manquants ne sont probablement pas aléatoires (jeunes
# n'ayant jamais bu, non-réponse…). Part pondérée de `Q19A` manquant par
# modalité des variables les plus parlantes.

# %%
na_vars = ["Q03", "Q05", "Q08C", "B08A", "B08B", "Q10A1"]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
global_na = np.average(y.isna(), weights=w) * 100
for ax, var in zip(axes.flat, na_vars):
    data = pd.DataFrame({"mod": as_labels(var), "na": y.isna(), "w": w})
    data["naw"] = data["na"] * data["w"]
    g = data.groupby("mod", observed=False).agg(naw=("naw", "sum"), w=("w", "sum"), n=("na", "size"))
    g = g[g["n"] >= 30]
    rate = g["naw"] / g["w"] * 100
    colors = [GREY if m == MISSING_LABEL else BLUE for m in rate.index]
    bars = ax.barh(rate.index.astype(str), rate.values, color=colors, height=0.7)
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7, color=TEXT)
    ax.axvline(global_na, color=TEXT, linestyle=":", linewidth=1)
    ax.invert_yaxis()
    ax.set_title(f"{var} · {VARIABLES[var][0]}")
    ax.grid(axis="y", visible=False)
for ax in axes[-1]:
    ax.set_xlabel("Q19A non renseigné (%, pondéré)")
fig.suptitle(f"Part de Q19A manquant par modalité (moyenne : {global_na:.0f} %)", fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()

# %% [markdown]
# ## 7. Liens entre prédicteurs : V de Cramér
#
# Calculé sur les effectifs pondérés, non-réponses incluses comme modalité.


# %%
def cramers_v(a: pd.Series, b: pd.Series, weights: pd.Series) -> float:
    table = pd.crosstab(a, b, values=weights, aggfunc="sum").fillna(0).to_numpy()
    table = table[table.sum(1) > 0][:, table.sum(0) > 0]
    n = table.sum()
    expected = np.outer(table.sum(1), table.sum(0)) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    k = min(table.shape) - 1
    return np.sqrt(chi2 / (n * k)) if k > 0 else np.nan


labelled = {v: pd.Series(as_labels(v)) for v in PREDICTORS}
V = pd.DataFrame(
    [[cramers_v(labelled[a], labelled[b], w) for b in PREDICTORS] for a in PREDICTORS],
    index=PREDICTORS,
    columns=PREDICTORS,
)

fig, ax = plt.subplots(figsize=(9, 7.5))
mask = np.triu(np.ones_like(V, dtype=bool))
vals = np.where(mask, np.nan, V.values)
cmap_v = LinearSegmentedColormap.from_list("blue_ramp", ["#ffffff"] + BLUE_RAMP)
im = ax.imshow(vals, cmap=cmap_v, vmin=0, vmax=1)
for i in range(len(V)):
    for j in range(i):
        ax.text(j, i, f"{V.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if V.values[i, j] > 0.5 else "#0b0b0b")
ax.set_xticks(range(len(V)), V.columns, rotation=45, ha="right")
ax.set_yticks(range(len(V)), V.index)
ax.set_title("V de Cramér entre prédicteurs (pondéré)")
ax.grid(False)
ax.spines[["left", "bottom"]].set_visible(False)
fig.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Premières conclusions
#
# - **Cible très concentrée** : environ 70 % des âges entre 14 et 16 ans (mode à 15),
#   moyenne pondérée de 14,3 ans. 88 valeurs hors de la plage [6, 18], dont des âges
#   impossibles (20 à 32 ans) : à nettoyer avant la modélisation.
# - **Consommation des parents = facteur le plus marqué** : l'âge moyen passe
#   d'environ 14,8 ans (aucun parent ne boit) à 13,4–13,8 ans (consommation
#   hebdomadaire ou quotidienne). Les deux variables sont corrélées (V ≈ 0,54).
# - **Effets plus faibles** : les garçons commencent plus tôt que les filles
#   (≈ 0,4 an d'écart), les enfants de cadres et d'agriculteurs aussi, tandis que le
#   redoublement et les difficultés de lecture/écriture vont avec un âge plus tardif.
# - **Les écarts entre groupes sont faibles** (moins d'1,5 an) comparés à la
#   dispersion (écart-type ≈ 2 ans) : il faut s'attendre à un R² modeste.
# - **Les `Q19A` manquants ne sont pas aléatoires** : 41 % quand le père ne boit
#   jamais, contre environ 10 % sinon. Ce sont probablement en partie des jeunes
#   n'ayant jamais bu, ce qui limite la population sur laquelle porte le modèle.
# - **Variables filtrées** : `Q04A` et `Q04B` dépendent de `Q04` (V ≈ 0,6) et sont
#   à fusionner en une seule variable « situation ».
