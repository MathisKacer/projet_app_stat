# %% [markdown]
# # 03 — Modèles de prédiction de l'âge au premier alcool
#
# Comparaison de méthodes d'agrégation (bagging, forêt aléatoire, boosting,
# stacking) avec des modèles de référence (moyenne, régression Ridge, arbre
# seul).
#
# ## Cible
#
# `Q19A` **continue** (le sujet demande de prédire une variable quantitative),
# restreinte aux jeunes **ayant déjà bu** : pour les 18 % de « Non concerné »,
# l'âge au premier alcool n'existe pas. `Q19A_CLASSE` sert uniquement à
# stratifier le découpage apprentissage / test (même répartition des classes
# d'âge dans les deux échantillons).
#
# ## Variables explicatives (voir `data/features.py`)
#
# | Variable | Contenu | Encodage | Pourquoi |
# |---|---|---|---|
# | `Q03` | Sexe | binaire | les garçons commencent plus tôt (≈ 0,4 an) |
# | `Q05` | Redoublement | binaire | lié à un âge plus tardif |
# | `Q06A`, `Q06B` | Difficultés lecture / écriture | ordinal 1-3 | idem, gradient visible |
# | `B08A`, `B08B` | Père / mère boit à la maison | ordinal 1-5 | effet le plus fort et monotone (14,8 → 13,4 ans) |
# | `SITUATION` | Études / arrêt × type | one-hot | résume `Q04`, `Q04A`, `Q04B` sans redondance |
# | `Q08` | Lieu de vie | one-hot | « propre logement » plus précoce |
# | `Q08C` | Vie des parents | one-hot | contexte familial |
# | `Q09A1`, `Q09B1` | Situation du père / de la mère | one-hot | contexte socio-économique |
# | `Q10A1`, `Q10B1` | PCS du père / de la mère | one-hot | enfants de cadres et d'agriculteurs plus précoces |
#
# - **Ordinales** : le score conserve l'ordre des modalités (un seul coefficient
#   pour la régression, des seuils naturels pour les arbres). « Non répondu »
#   est imputé par la médiane **et** signalé par une indicatrice, car la
#   non-réponse n'est pas aléatoire.
# - **Nominales** : indicatrices ; « Non répondu » et « Non concerné » sont des
#   modalités à part entière ; les modalités de moins de 30 individus sont
#   regroupées.
# - **Exclues** : `A01` (identifiant), `pm17B` (poids de sondage : utilisé comme
#   pondération à l'apprentissage et dans les métriques, pas comme prédicteur),
#   `Q04`, `Q04A`, `Q04B` (remplacées par `SITUATION`), `Q19A_CLASSE` (dérivée
#   de la cible : fuite d'information).
#
# ## Protocole
#
# - Découpage 80 % apprentissage / 20 % test, stratifié sur `Q19A_CLASSE`.
# - Hyperparamètres choisis par validation croisée à 5 plis sur l'apprentissage.
# - Apprentissage pondéré par `pm17B` ; métriques pondérées (RMSE, MAE, R²).
# - Le test n'est utilisé qu'une fois, pour la comparaison finale.

# %%
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

ROOT = Path.cwd() if (Path.cwd() / "data").is_dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from data import load_escap_data, preprocess  # noqa: E402
from data.features import FEATURES, make_preprocessor  # noqa: E402
from data.labels import TARGET, VARIABLES, WEIGHT  # noqa: E402
from data.preprocess import TARGET_CLASS  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

SEED = 42
N_JOBS = -1

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8985"
TEXT = "#52514e"

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
# ## Données

# %%
df = preprocess(load_escap_data())
df = df[df[TARGET].notna()].reset_index(drop=True)

X, y, w = df[FEATURES], df[TARGET], df[WEIGHT]
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, stratify=df[TARGET_CLASS], random_state=SEED
)
strata_train = df.loc[X_train.index, TARGET_CLASS]

print(f"Jeunes ayant déjà bu : {len(df)}")
print(f"Apprentissage : {len(X_train)} — test : {len(X_test)}")
print(f"Variables : {len(FEATURES)} brutes -> "
      f"{make_preprocessor().fit(X_train).get_feature_names_out().size} colonnes encodées")

# %% [markdown]
# ## Outils : validation croisée pondérée et métriques


# %%
def weighted_rmse(y_true, y_pred, weights):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=weights))


def make_pipe(estimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])


CV = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_train, strata_train))


def _fit_score(estimator, params, train_idx, val_idx):
    pipe = make_pipe(clone(estimator).set_params(**params))
    pipe.fit(
        X_train.iloc[train_idx], y_train.iloc[train_idx],
        model__sample_weight=w_train.iloc[train_idx].to_numpy(),
    )
    pred = pipe.predict(X_train.iloc[val_idx])
    return weighted_rmse(y_train.iloc[val_idx], pred, w_train.iloc[val_idx])


def grid_search(estimator, grid) -> pd.DataFrame:
    """RMSE pondérée en validation croisée pour chaque combinaison d'hyperparamètres."""
    combos = list(ParameterGrid(grid))
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_score)(estimator, p, tr, va) for p in combos for tr, va in CV
    )
    scores = np.asarray(scores).reshape(len(combos), len(CV))
    res = pd.DataFrame(combos)
    res["rmse_cv"], res["rmse_cv_std"] = scores.mean(1), scores.std(1)
    res["params"] = combos
    return res.sort_values("rmse_cv").reset_index(drop=True)


RESULTS = {}
FITTED = {}


def evaluate(name, estimator, grid=None):
    """Choisit les hyperparamètres en CV, réapprend sur tout l'apprentissage, évalue sur le test."""
    start = time.time()
    search = grid_search(estimator, grid or {})
    best = search.iloc[0]
    params = best["params"]
    pipe = make_pipe(clone(estimator).set_params(**params))
    pipe.fit(X_train, y_train, model__sample_weight=w_train.to_numpy())
    pred = pipe.predict(X_test)
    FITTED[name] = pipe
    RESULTS[name] = {
        "RMSE CV": best["rmse_cv"],
        "± CV": best["rmse_cv_std"],
        "RMSE test": weighted_rmse(y_test, pred, w_test),
        "MAE test": mean_absolute_error(y_test, pred, sample_weight=w_test),
        "R² test": r2_score(y_test, pred, sample_weight=w_test),
        "temps (s)": time.time() - start,
        "hyperparamètres": params,
    }
    print(f"{name} — meilleurs hyperparamètres : {params}")
    print(pd.Series(RESULTS[name]).drop("hyperparamètres").round(4).to_string())
    return search


# %% [markdown]
# ## Modèles de référence
#
# - **Moyenne pondérée** : ce qu'il faut battre (RMSE ≈ écart-type de la cible).
# - **Ridge** : modèle linéaire pénalisé, référence interprétable.
# - **Arbre de décision** seul : la brique de base des méthodes d'agrégation,
#   instable et vite en sur-apprentissage.

# %%
evaluate("Moyenne", DummyRegressor(strategy="mean"))

# %%
search_ridge = evaluate("Ridge", Ridge(), {"alpha": [float(a) for a in np.logspace(-1, 4, 11)]})

# %%
search_tree = evaluate(
    "Arbre",
    DecisionTreeRegressor(random_state=SEED),
    {"max_depth": [2, 3, 4, 5, 6, 8, None], "min_samples_leaf": [30, 100, 200, 400]},
)
search_tree.drop(columns="params").head()

# %% [markdown]
# ## Bagging
#
# Moyenne de 300 arbres profonds appris sur des échantillons bootstrap : réduit
# la variance de l'arbre seul. On règle la taille des feuilles et la taille des
# échantillons bootstrap.

# %%
search_bag = evaluate(
    "Bagging",
    BaggingRegressor(
        DecisionTreeRegressor(), n_estimators=300, random_state=SEED, n_jobs=1
    ),
    {"estimator__min_samples_leaf": [5, 20, 50, 100, 200], "max_samples": [0.2, 0.5, 1.0]},
)
search_bag.drop(columns="params").head()

# %% [markdown]
# ## Forêt aléatoire
#
# Bagging + tirage aléatoire des variables à chaque nœud (`max_features`), ce
# qui décorrèle les arbres.

# %%
search_rf = evaluate(
    "Forêt aléatoire",
    RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
    {"max_features": [0.1, 0.2, 0.33, 0.5], "min_samples_leaf": [5, 10, 20, 50]},
)
search_rf.drop(columns="params").head()

# %% [markdown]
# ### Nombre d'arbres : bagging vs forêt aléatoire
#
# Erreur *out-of-bag* pondérée (individus non tirés dans le bootstrap) en
# fonction du nombre d'arbres, avec les meilleurs hyperparamètres.

# %%
prep = make_preprocessor().fit(X_train)
Xt_train = prep.transform(X_train)
n_trees = [25, 50, 100, 200, 400]
oob = {"Bagging": [], "Forêt aléatoire": []}
for n in n_trees:
    for name in oob:
        model = clone(FITTED[name].named_steps["model"]).set_params(
            n_estimators=n, oob_score=True, n_jobs=N_JOBS
        )
        with warnings.catch_warnings():  # quelques individus sans prédiction OOB à 25 arbres
            warnings.simplefilter("ignore", UserWarning)
            model.fit(Xt_train, y_train, sample_weight=w_train.to_numpy())
        pred = model.oob_prediction_
        ok = ~np.isnan(pred)
        oob[name].append(weighted_rmse(y_train[ok], pred[ok], w_train[ok]))

fig, ax = plt.subplots(figsize=(7, 3.5))
for (name, values), color in zip(oob.items(), (GREY, BLUE)):
    ax.plot(n_trees, values, marker="o", color=color, linewidth=2, markersize=5,
            markeredgecolor="white", label=name)
ax.set_xscale("log")
ax.set_xticks(n_trees, n_trees)
ax.minorticks_off()
ax.set_xlabel("Nombre d'arbres")
ax.set_ylabel("RMSE out-of-bag (pondérée)")
ax.set_title("Erreur OOB selon le nombre d'arbres")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Boosting
#
# Arbres peu profonds ajoutés séquentiellement, chacun corrigeant les résidus
# des précédents. Compromis clé : taux d'apprentissage × nombre d'itérations.
#
# - `GradientBoostingRegressor` : implémentation classique, avec
#   sous-échantillonnage (boosting stochastique).
# - `HistGradientBoostingRegressor` : version par histogrammes (type LightGBM),
#   plus rapide, avec régularisation L2.

# %%
search_gb = evaluate(
    "Gradient boosting",
    GradientBoostingRegressor(subsample=0.8, random_state=SEED),
    {
        "learning_rate": [0.005, 0.01, 0.02, 0.05],
        "n_estimators": [300, 1000],
        "max_depth": [1, 2, 3],
    },
)
search_gb.drop(columns="params").head()

# %%
search_hgb = evaluate(
    "Hist. gradient boosting",
    HistGradientBoostingRegressor(early_stopping=False, random_state=SEED),
    {
        "learning_rate": [0.01, 0.03, 0.1],
        "max_iter": [100, 300, 600],
        "max_leaf_nodes": [3, 7, 15],
        "min_samples_leaf": [20, 80],
        "l2_regularization": [0.0, 1.0],
    },
)
search_hgb.drop(columns="params").head()

# %% [markdown]
# ### Taux d'apprentissage et nombre d'itérations
#
# Erreur d'apprentissage et de validation (20 % de l'apprentissage mis de côté)
# au fil des itérations : un petit taux demande plus d'itérations mais
# sur-apprend moins vite.

# %%
Xt_fit, Xt_val, y_fit, y_val, w_fit, w_val = train_test_split(
    Xt_train, y_train, w_train, test_size=0.2, random_state=SEED
)
fig, ax = plt.subplots(figsize=(8, 4))
for lr, color in ((0.02, BLUE), (0.1, ORANGE), (0.5, GREY)):
    gb = GradientBoostingRegressor(
        learning_rate=lr, n_estimators=800, max_depth=3, subsample=0.8, random_state=SEED
    ).fit(Xt_fit, y_fit, sample_weight=w_fit.to_numpy())
    val = [weighted_rmse(y_val, p, w_val) for p in gb.staged_predict(Xt_val)]
    train = [weighted_rmse(y_fit, p, w_fit) for p in gb.staged_predict(Xt_fit)]
    ax.plot(val, color=color, linewidth=2, label=f"validation, lr = {lr}")
    ax.plot(train, color=color, linewidth=1, linestyle="--", label=f"apprentissage, lr = {lr}")
ax.set_xlabel("Nombre d'itérations")
ax.set_ylabel("RMSE pondérée")
ax.set_title("Gradient boosting : sur-apprentissage selon le taux d'apprentissage")
ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
ax.set_ylim(1.7, 2.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Stacking
#
# Combine les prédictions de modèles différents (Ridge, forêt aléatoire,
# boosting par histogrammes, avec leurs meilleurs hyperparamètres). Les
# prédictions des modèles de base sont obtenues par validation croisée interne,
# puis une régression linéaire à coefficients positifs apprend leur poids.

# %%
base = [
    ("ridge", FITTED["Ridge"].named_steps["model"]),
    ("rf", FITTED["Forêt aléatoire"].named_steps["model"]),
    ("hgb", FITTED["Hist. gradient boosting"].named_steps["model"]),
]
evaluate(
    "Stacking",
    StackingRegressor(
        [(n, clone(m)) for n, m in base],
        final_estimator=LinearRegression(positive=True),
        cv=5,
        n_jobs=1,
    ),
)
stack = FITTED["Stacking"].named_steps["model"]
print("Poids du méta-modèle :",
      dict(zip([n for n, _ in base], stack.final_estimator_.coef_.round(3))),
      f"constante = {stack.final_estimator_.intercept_:.3f}")

# %% [markdown]
# ## Comparaison des modèles

# %%
results = pd.DataFrame(RESULTS).T
results_num = results.drop(columns="hyperparamètres").astype(float)
results_num.sort_values("RMSE test").round(4)

# %%
order = results_num.sort_values("RMSE test", ascending=False)
best_name = order.index[-1]
REFERENCES = ("Moyenne", "Ridge", "Arbre")
best_ensemble = next(n for n in order.index[::-1] if n not in REFERENCES)
print(f"Meilleur modèle : {best_name} — meilleur modèle d'agrégation : {best_ensemble}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
colors = [GREY if n in REFERENCES else BLUE for n in order.index]
ax1.barh(order.index, order["RMSE test"], color=colors, height=0.6)
ax1.errorbar(order["RMSE CV"], np.arange(len(order)), xerr=order["± CV"], fmt="o",
             color=TEXT, markersize=4, elinewidth=1, label="RMSE CV (± écart-type)")
for i, v in enumerate(order["RMSE test"]):
    ax1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
ax1.set_xlim(order["RMSE test"].min() - 0.15, order["RMSE test"].max() + 0.1)
ax1.set_xlabel("RMSE pondérée (ans) — barres : test")
ax1.set_title("Erreur de prédiction")
ax1.legend(loc="upper right")
ax1.grid(axis="y", visible=False)

bars = ax2.barh(order.index, order["R² test"] * 100, color=colors, height=0.6)
ax2.bar_label(bars, fmt="%.1f %%", padding=3, fontsize=8, color=TEXT)
ax2.set_xlabel("R² test (%, pondéré)")
ax2.set_title("Part de variance expliquée")
ax2.set_xlim(min(0, order["R² test"].min() * 100) - 1, order["R² test"].max() * 100 * 1.3)
ax2.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Interprétation
#
# ### Importance des variables (permutation, sur le test)
#
# Augmentation de la RMSE pondérée quand on permute une variable : mesure ce
# que le modèle perd sans l'information de cette variable.

# %%
imp = {}
for name in ("Ridge", best_ensemble):
    r = permutation_importance(
        FITTED[name], X_test, y_test, sample_weight=w_test.to_numpy(),
        scoring="neg_root_mean_squared_error", n_repeats=20, random_state=SEED, n_jobs=N_JOBS,
    )
    imp[name] = pd.Series(r.importances_mean, index=FEATURES)
imp = pd.DataFrame(imp).sort_values(best_ensemble)
labels = [f"{v} · {VARIABLES[v][0]}" if v in VARIABLES else f"{v} · Situation (synthèse)"
          for v in imp.index]

fig, ax = plt.subplots(figsize=(8, 5))
pos = np.arange(len(imp))
ax.barh(pos + 0.2, imp[best_ensemble], height=0.38, color=BLUE, label=best_ensemble)
ax.barh(pos - 0.2, imp["Ridge"], height=0.38, color=GREY, label="Ridge")
ax.set_yticks(pos, labels)
ax.axvline(0, color=TEXT, linewidth=0.8)
ax.set_xlabel("Hausse de la RMSE quand la variable est permutée (ans)")
ax.set_title("Importance des variables par permutation")
ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Prédictions vs valeurs observées
#
# Prédiction moyenne du meilleur modèle d'agrégation pour chaque âge observé, avec
# l'intervalle interquartile des prédictions.

# %%
pred_best = FITTED[best_ensemble].predict(X_test)
by_age = (
    pd.DataFrame({"obs": y_test, "pred": pred_best})
    .groupby("obs")["pred"]
    .agg(["mean", lambda p: p.quantile(0.25), lambda p: p.quantile(0.75), "size"])
    .set_axis(["mean", "q1", "q3", "n"], axis=1)
)
by_age = by_age[by_age["n"] >= 10]

fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(by_age.index, by_age["q1"], by_age["q3"], color="#cde2fb",
                label="Prédictions : intervalle interquartile")
ax.plot(by_age.index, by_age["mean"], marker="o", color=BLUE, linewidth=2, markersize=5,
        markeredgecolor="white", label="Prédiction moyenne")
lims = [by_age.index.min(), by_age.index.max()]
ax.plot(lims, lims, color=TEXT, linestyle=":", linewidth=1, label="Prédiction parfaite")
ax.set_xticks(by_age.index.astype(int))
ax.set_xlabel("Âge observé (ans)")
ax.set_ylabel("Âge prédit (ans)")
ax.set_title(f"{best_ensemble} : prédictions sur le test")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

print(f"Plage des prédictions : [{pred_best.min():.1f}, {pred_best.max():.1f}] ans "
      f"(observé : [{y_test.min():.0f}, {y_test.max():.0f}])")

# %% [markdown]
# ## Conclusions
#
# - **Pouvoir prédictif faible pour tous les modèles** : R² test ≈ 4-5 %, RMSE
#   ≈ 2,10 ans contre 2,16 pour la simple moyenne. Les prédictions restent
#   entre 13 et 16 ans environ : les variables disponibles (situation
#   personnelle, contexte familial) n'expliquent qu'une petite partie de l'âge au
#   premier alcool, ce qui confirme la statistique descriptive (écarts entre
#   groupes < 1,5 an pour un écart-type ≈ 2 ans).
# - **Les méthodes d'agrégation font mieux que l'arbre seul** (bagging et forêt
#   réduisent sa variance) **mais pas mieux que Ridge** : les écarts entre Ridge,
#   boosting et stacking (≈ 0,002 an) sont bien inférieurs à l'écart-type entre
#   plis de validation croisée (≈ 0,03).
# - **Structure additive** : le boosting retient des arbres à une seule
#   coupure (`max_depth = 1`, `max_leaf_nodes = 3`), c'est-à-dire un modèle
#   additif sans interactions, que la régression linéaire capte déjà.
# - **Bagging vs forêt** : la forêt a besoin de plus d'arbres pour se stabiliser
#   mais finit légèrement meilleure (arbres décorrélés par `max_features`) ;
#   au-delà de 200 arbres, le gain est nul.
# - **Boosting** : un taux d'apprentissage élevé sur-apprend très vite ; un taux
#   faible donne une erreur de validation stable avec plus d'itérations.
# - **Stacking** : la combinaison donne surtout du poids à Ridge et à la forêt,
#   sans gain notable, car les modèles de base font des erreurs très corrélées.
# - **Variables** : le sexe et la consommation d'alcool des parents dominent,
#   puis les PCS des parents et le redoublement ; Ridge et le modèle
#   d'agrégation s'accordent sur ce classement.
