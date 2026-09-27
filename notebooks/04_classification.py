# %% [markdown]
# # 04 — Classification de l'âge au premier alcool
#
# On prédit une version discrétisée de l'âge au premier alcool, en **regroupant
# 14 et 15 ans** :
#
# | Classe | Contenu |
# |---|---|
# | ≤ 13 ans | initiation précoce |
# | 14-15 ans | âge le plus courant |
# | ≥ 16 ans | initiation tardive |
# | Non concerné | n'a jamais bu (`Q19A` manquant) |
#
# Pourquoi ce regroupement : avec le découpage en quartiles (≤ 13 / 14 / 15 /
# ≥ 16), les modèles ne distinguaient pas les jeunes de 14 et 15 ans (mêmes
# probabilités prédites, classe « 14 ans » jamais prédite). Les seuils (13 et
# 15) sont donc fixés à la main et non plus par quantiles. Contrepartie : la
# classe 14-15 ans regroupe environ 41 % des jeunes ; les scores (exactitude
# notamment) ne sont pas comparables tels quels avec la version à 5 classes, on
# les lit par rapport à la référence « classe majoritaire ».
#
# Par rapport à la régression (`03_modeles.py`) :
#
# - **tous les jeunes sont modélisés**, y compris ceux qui n'ont jamais bu ;
# - les âges extrêmes (1 à 5 ans) ne pèsent plus plus que les autres ;
# - le problème devient : reconnaître les profils précoces, tardifs ou abstinents.
#
# ## Variables explicatives
#
# Les mêmes que pour la régression (`data/features.py`), pour les mêmes raisons :
# sexe, redoublement, difficultés de lecture / écriture, consommation d'alcool
# des parents (scores ordinaux + indicatrice de non-réponse), `SITUATION`, lieu
# de vie, vie des parents, situation et PCS des parents (indicatrices).
# Exclues : `A01` (identifiant), `pm17B` (pondération), `Q04`/`Q04A`/`Q04B`
# (résumées par `SITUATION`), `Q19A` (la cible continue : fuite d'information).
#
# ## Protocole
#
# - Découpage 80 % apprentissage / 20 % test, stratifié sur la classe.
# - Hyperparamètres choisis par validation croisée à 5 plis en minimisant la
#   **log-loss pondérée** : elle évalue les probabilités prédites, ce qui est plus
#   stable que l'exactitude quand le signal est faible (les modèles tendent à
#   prédire toujours les mêmes classes).
# - Apprentissage et métriques pondérés par `pm17B` : log-loss, exactitude,
#   F1 macro (moyenne des F1 de chaque classe, sensible aux classes peu
#   prédites) et AUC macro One-vs-Rest (moyenne des AUC « classe k contre les
#   autres » : capacité des probabilités à classer les individus, sans dépendre
#   de la classe prédite).

# %%
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path.cwd() if (Path.cwd() / "data").is_dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from data import load_escap_data, preprocess  # noqa: E402
from data.features import FEATURES, make_preprocessor  # noqa: E402
from data.labels import VARIABLES, WEIGHT  # noqa: E402
from data.preprocess import TARGET_CLASS  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

SEED = 42
N_JOBS = -1

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8985"
TEXT = "#52514e"
BLUE_RAMP = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

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
THRESHOLDS = [13, 15]  # classes : ≤ 13, 14-15, ≥ 16 ans
df = preprocess(load_escap_data(), thresholds=THRESHOLDS)
CLASSES = list(df[TARGET_CLASS].cat.categories)

X, y, w = df[FEATURES], df[TARGET_CLASS].astype(str), df[WEIGHT]
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, stratify=y, random_state=SEED
)

print(f"Individus : {len(df)} — apprentissage : {len(X_train)}, test : {len(X_test)}")
shares = w.groupby(y).sum().reindex(CLASSES) / w.sum() * 100
pd.DataFrame({"effectif": y.value_counts().reindex(CLASSES), "% pondéré": shares.round(1)})

# %% [markdown]
# ## Outils : validation croisée pondérée et métriques


# %%
def make_pipe(estimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])


def weighted_log_loss(pipe, X_, y_, w_):
    return log_loss(y_, pipe.predict_proba(X_), sample_weight=w_, labels=pipe.classes_)


CV = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_train, y_train))


def _fit_score(estimator, params, train_idx, val_idx):
    pipe = make_pipe(clone(estimator).set_params(**params))
    pipe.fit(
        X_train.iloc[train_idx], y_train.iloc[train_idx],
        model__sample_weight=w_train.iloc[train_idx].to_numpy(),
    )
    return weighted_log_loss(
        pipe, X_train.iloc[val_idx], y_train.iloc[val_idx], w_train.iloc[val_idx]
    )


def grid_search(estimator, grid) -> pd.DataFrame:
    """Log-loss pondérée en validation croisée pour chaque combinaison d'hyperparamètres."""
    combos = list(ParameterGrid(grid))
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_score)(estimator, p, tr, va) for p in combos for tr, va in CV
    )
    scores = np.asarray(scores).reshape(len(combos), len(CV))
    res = pd.DataFrame(combos)
    res["logloss_cv"], res["logloss_cv_std"] = scores.mean(1), scores.std(1)
    res["params"] = combos
    return res.sort_values("logloss_cv").reset_index(drop=True)


RESULTS = {}
FITTED = {}


def evaluate(name, estimator, grid=None):
    """Choisit les hyperparamètres en CV, réapprend sur tout l'apprentissage, évalue sur le test."""
    start = time.time()
    search = grid_search(estimator, grid or {})
    best = search.iloc[0]
    pipe = make_pipe(clone(estimator).set_params(**best["params"]))
    pipe.fit(X_train, y_train, model__sample_weight=w_train.to_numpy())
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    FITTED[name] = pipe
    RESULTS[name] = {
        "log-loss CV": best["logloss_cv"],
        "± CV": best["logloss_cv_std"],
        "log-loss test": weighted_log_loss(pipe, X_test, y_test, w_test),
        "exactitude test": accuracy_score(y_test, pred, sample_weight=w_test),
        "F1 macro test": f1_score(y_test, pred, average="macro", sample_weight=w_test),
        "AUC macro test": roc_auc_score(y_test, proba, multi_class="ovr", average="macro",
                                        sample_weight=w_test, labels=pipe.classes_),
        "temps (s)": time.time() - start,
        "hyperparamètres": best["params"],
    }
    print(f"{name} — meilleurs hyperparamètres : {best['params']}")
    print(pd.Series(RESULTS[name]).drop("hyperparamètres").round(4).to_string())
    return search.drop(columns="params")


# %% [markdown]
# ## Modèles de référence
#
# - **Classe majoritaire** (probabilités = fréquences des classes) : ce qu'il faut
#   battre.
# - **Régressions logistiques multinomiales pénalisées**, `C` étant l'inverse de
#   la force de pénalité :
#   - **ridge** (L2) : réduit tous les coefficients sans en annuler ;
#   - **lasso** (L1) : annule une partie des coefficients (sélection de
#     variables) ;
#   - **elastic net** : mélange des deux, `l1_ratio` étant la part de L1.
#
#   Les variables ne sont pas standardisées (scores ordinaux de 1 à k et
#   indicatrices 0/1, d'échelles proches).
# - **Arbre de décision** seul.

# %%
evaluate("Classe majoritaire", DummyClassifier(strategy="prior"))

# %%
evaluate(
    "Logistique ridge",
    LogisticRegression(max_iter=5000),
    {"C": [float(c) for c in np.logspace(-3, 1, 9)]},
)

# %%
evaluate(
    "Logistique lasso",
    LogisticRegression(l1_ratio=1.0, solver="saga", max_iter=5000),
    {"C": [float(c) for c in np.logspace(-3, 1, 9)]},
)

# %%
evaluate(
    "Logistique elastic net",
    LogisticRegression(solver="saga", max_iter=5000),
    {"C": [float(c) for c in np.logspace(-3, 1, 9)], "l1_ratio": [0.25, 0.5, 0.75]},
).head()

# %% [markdown]
# Nombre de coefficients non nuls (pour au moins une classe) et variables
# entièrement écartées par le lasso et l'elastic net.

# %%
LOGITS = ("Logistique ridge", "Logistique lasso", "Logistique elastic net")
coef_names = FITTED["Logistique ridge"].named_steps["prep"].get_feature_names_out()
nonzero = pd.DataFrame(
    {n: np.abs(FITTED[n].named_steps["model"].coef_).max(0) > 1e-10 for n in LOGITS},
    index=coef_names,
)
print(nonzero.sum().to_string(), f"\nsur {len(coef_names)} coefficients par classe")
nonzero[~nonzero.all(axis=1)]

# %%
evaluate(
    "Arbre",
    DecisionTreeClassifier(random_state=SEED),
    {"max_depth": [3, 4, 6, 8, 10, 12], "min_samples_leaf": [50, 100, 200, 400]},
).head()

# %% [markdown]
# ## Bagging et forêt aléatoire

# %%
evaluate(
    "Bagging",
    BaggingClassifier(DecisionTreeClassifier(), n_estimators=300, random_state=SEED, n_jobs=1),
    {"estimator__min_samples_leaf": [5, 10, 20, 50, 100], "max_samples": [0.2, 0.5, 1.0]},
).head()

# %%
evaluate(
    "Forêt aléatoire",
    RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
    {"max_features": [0.1, 0.2, 0.33, 0.5], "min_samples_leaf": [10, 20, 50, 100]},
).head()

# %% [markdown]
# ## Boosting

# %%
evaluate(
    "Gradient boosting",
    GradientBoostingClassifier(subsample=0.8, random_state=SEED),
    {
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "n_estimators": [100, 300, 600],
        "max_depth": [1, 2, 3],
    },
).head()

# %%
evaluate(
    "Hist. gradient boosting",
    HistGradientBoostingClassifier(early_stopping=False, random_state=SEED),
    {
        "learning_rate": [0.01, 0.03, 0.1],
        "max_iter": [100, 300],
        "max_leaf_nodes": [3, 7, 15],
        "min_samples_leaf": [20, 80],
        "l2_regularization": [0.0, 1.0],
    },
).head()

# %% [markdown]
# ### XGBoost
#
# XGBoost exige des classes codées 0, 1, …, K-1 : la classe ci-dessous code et
# décode les libellés pour s'utiliser comme les autres modèles. Même grille que
# le gradient boosting, avec en plus le sous-échantillonnage des variables
# (`colsample_bytree`).


# %%
class XGBLabelClassifier(XGBClassifier):
    """XGBClassifier acceptant des classes textuelles."""

    @property
    def classes_(self):
        # pendant le fit, XGBoost compare classes_ aux codes 0..K-1
        le = getattr(self, "label_encoder_", None)
        return le.classes_ if le is not None else np.arange(self.n_classes_)

    def fit(self, X, y, **kwargs):
        self.label_encoder_ = None
        le = LabelEncoder().fit(y)
        super().fit(X, le.transform(y), **kwargs)
        self.label_encoder_ = le
        return self

    def predict(self, X):
        return self.label_encoder_.inverse_transform(super().predict(X))


# %%
evaluate(
    "XGBoost",
    XGBLabelClassifier(subsample=0.8, tree_method="hist", n_jobs=1, random_state=SEED),
    {
        "learning_rate": [0.02, 0.05, 0.1],
        "n_estimators": [100, 300, 600],
        "max_depth": [1, 2, 3],
        "colsample_bytree": [0.8, 1.0],
    },
).head()

# %% [markdown]
# ## Stacking
#
# Régression logistique, forêt aléatoire et boosting par histogrammes (meilleurs
# hyperparamètres) ; leurs probabilités prédites (par validation croisée
# interne) alimentent une régression logistique.

# %%
base = [
    ("logit", FITTED["Logistique ridge"].named_steps["model"]),
    ("rf", FITTED["Forêt aléatoire"].named_steps["model"]),
    ("hgb", FITTED["Hist. gradient boosting"].named_steps["model"]),
]
evaluate(
    "Stacking",
    StackingClassifier(
        [(n, clone(m)) for n, m in base],
        final_estimator=LogisticRegression(max_iter=5000),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
    ),
)

# %% [markdown]
# ## Comparaison des modèles

# %%
results = pd.DataFrame(RESULTS).T
results_num = results.drop(columns="hyperparamètres").astype(float)
baseline = results_num.loc["Classe majoritaire", "log-loss test"]
results_num["gain log-loss vs référence (%)"] = (1 - results_num["log-loss test"] / baseline) * 100
results_num.sort_values("log-loss test").round(4)

# %%
REFERENCES = ("Classe majoritaire", *LOGITS, "Arbre")
order = results_num.sort_values("log-loss test", ascending=False)
best_name = order.index[-1]
best_ensemble = next(n for n in order.index[::-1] if n not in REFERENCES)
print(f"Meilleur modèle : {best_name} — meilleur modèle d'agrégation : {best_ensemble}")

colors = [GREY if n in REFERENCES else BLUE for n in order.index]
pos = np.arange(len(order))
fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
for ax, metric in zip(axes, ("log-loss test", "exactitude test", "F1 macro test",
                             "AUC macro test")):
    ax.hlines(pos, order[metric].min(), order[metric], color="#e4e3df", linewidth=1)
    ax.scatter(order[metric], pos, color=colors, s=45, zorder=3, edgecolor="white")
    for p_, v in zip(pos, order[metric]):
        ax.annotate(f"{v:.3f}", (v, p_), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7, color=TEXT)
    sens = "plus bas = meilleur" if "loss" in metric else "plus haut = meilleur"
    ax.set_title(f"{metric.replace(' test', '').capitalize()} ({sens})")
    ax.set_yticks(pos, order.index)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.grid(axis="y", visible=False)
axes[0].errorbar(order["log-loss CV"], pos - 0.25, xerr=order["± CV"], fmt="o", color=GREY,
                 markerfacecolor="white", markersize=4, elinewidth=1,
                 label="validation croisée (± écart-type)")
axes[0].legend(loc="center right")
fig.suptitle("Performances sur le test (pondérées)", fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

# %% [markdown]
# ## Analyse du meilleur modèle d'agrégation
#
# ### Matrice de confusion
#
# En ligne la classe observée, en colonne la classe prédite ; chaque ligne
# somme à 100 % (pondéré).

# %%
pred = FITTED[best_ensemble].predict(X_test)
cm = confusion_matrix(y_test, pred, labels=CLASSES, sample_weight=w_test)
cm_pct = cm / cm.sum(1, keepdims=True) * 100

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(cm_pct, cmap=LinearSegmentedColormap.from_list("ramp", BLUE_RAMP), vmin=0,
               vmax=100)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, f"{cm_pct[i, j]:.0f} %", ha="center", va="center", fontsize=8,
                color="white" if cm_pct[i, j] > 55 else "#0b0b0b")
ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
ax.set_yticks(range(len(CLASSES)), CLASSES)
ax.set_xlabel("Classe prédite")
ax.set_ylabel("Classe observée")
ax.set_title(f"{best_ensemble} : matrice de confusion (test)")
ax.grid(False)
fig.colorbar(im, ax=ax, label="% de la classe observée", shrink=0.8)
plt.tight_layout()
plt.show()

pred_share = w_test.groupby(pd.Series(pred, index=y_test.index)).sum() / w_test.sum() * 100
pd.DataFrame(
    {
        "% observé": (w_test.groupby(y_test).sum() / w_test.sum() * 100),
        "% prédit": pred_share,
        "F1": pd.Series(
            f1_score(y_test, pred, labels=CLASSES, average=None, sample_weight=w_test),
            index=CLASSES,
        ),
    }
).reindex(CLASSES).fillna(0).round(3)

# %% [markdown]
# ### Probabilités prédites selon la classe observée
#
# Probabilité moyenne attribuée à chaque classe : un modèle utile donne plus
# de probabilité à la bonne classe (diagonale) que la fréquence de base.

# %%
proba = pd.DataFrame(FITTED[best_ensemble].predict_proba(X_test),
                     columns=FITTED[best_ensemble].classes_, index=y_test.index)[CLASSES]
mean_proba = proba.mul(w_test, axis=0).groupby(y_test).sum().div(
    w_test.groupby(y_test).sum(), axis=0
).reindex(CLASSES) * 100
base_rate = w_train.groupby(y_train).sum().reindex(CLASSES) / w_train.sum() * 100
mean_proba.loc["Fréquence de base"] = base_rate
mean_proba.round(1)

# %% [markdown]
# ### AUC par classe (One-vs-Rest)
#
# Pour chaque classe, probabilité qu'un jeune de cette classe reçoive une
# probabilité prédite plus élevée pour elle qu'un jeune d'une autre classe
# (0,5 = hasard). Contrairement au F1, elle ne dépend pas de la classe prédite :
# elle mesure le signal même pour les classes jamais prédites.

# %%
auc_class = pd.DataFrame(
    {
        name: [
            roc_auc_score(y_test == c, FITTED[name].predict_proba(X_test)[:, i],
                          sample_weight=w_test)
            for i, c in enumerate(FITTED[name].classes_)
        ]
        for name in ("Logistique ridge", "XGBoost", best_ensemble)
    },
    index=FITTED[best_ensemble].classes_,
).reindex(CLASSES)
auc_class.loc["Macro"] = auc_class.mean()
auc_class.round(3)

# %% [markdown]
# ### Importance des variables (permutation, sur le test)
#
# Hausse de la log-loss pondérée quand on permute une variable.

# %%
imp = {}
for name in ("Logistique ridge", best_ensemble):
    r = permutation_importance(
        FITTED[name], X_test, y_test, sample_weight=w_test.to_numpy(),
        scoring="neg_log_loss", n_repeats=20, random_state=SEED, n_jobs=N_JOBS,
    )
    imp[name] = pd.Series(r.importances_mean, index=FEATURES)
imp = pd.DataFrame(imp).sort_values(best_ensemble)
labels = [f"{v} · {VARIABLES[v][0]}" if v in VARIABLES else f"{v} · Situation (synthèse)"
          for v in imp.index]

fig, ax = plt.subplots(figsize=(8, 5))
pos = np.arange(len(imp))
ax.barh(pos + 0.2, imp[best_ensemble], height=0.38, color=BLUE, label=best_ensemble)
ax.barh(pos - 0.2, imp["Logistique ridge"], height=0.38, color=GREY,
        label="Logistique ridge")
ax.set_yticks(pos, labels)
ax.axvline(0, color=TEXT, linewidth=0.8)
ax.set_xlabel("Hausse de la log-loss quand la variable est permutée")
ax.set_title("Importance des variables par permutation")
ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusions
#
# - **Gain par rapport à la référence** : la log-loss passe de 1,326 (fréquences
#   des classes) à 1,232 pour le stacking, soit un gain de 7,0 % (contre 6,3 %
#   avec les 5 classes 14 / 15 séparées). Le regroupement améliore donc un peu la
#   qualité des probabilités.
# - **Mais la classe 14-15 ans absorbe presque toutes les prédictions** : environ
#   82 % des jeunes du test sont classés « 14-15 ans ». « ≤ 13 ans » et
#   « ≥ 16 ans » ne sont presque jamais prédits (F1 ≈ 0,04) ; seule la classe
#   « Non concerné » reste reconnue (43 % bien classés, F1 ≈ 0,47).
#   L'exactitude (45 %) ne dépasse la référence « toujours 14-15 ans » (41 %)
#   que de 4 points.
# - **Lecture** : aucun profil n'est assez marqué pour que le modèle donne plus de
#   probabilité à « ≤ 13 ans » ou à « ≥ 16 ans » qu'à la classe majoritaire. Le
#   signal existe (probabilités mieux calibrées que la référence) mais il est
#   trop faible pour changer la classe prédite, sauf pour les abstinents.
# - **AUC** : l'AUC macro One-vs-Rest n'est que de 0,65 pour le stacking (0,50
#   pour la référence). Par classe, les abstinents sont les mieux repérés (0,74),
#   puis les précoces (0,67) ; les classes « 14-15 ans » (0,61) et « ≥ 16 ans »
#   (0,59) sont à peine mieux séparées qu'au hasard. Le signal sur les précoces
#   existe donc, même si la classe n'est presque jamais prédite.
# - **Modèles** : comme avec 5 classes, les méthodes d'agrégation font un peu
#   mieux que les régressions logistiques (gain de 6,6 à 7,0 % contre 5,9 %), le
#   stacking en tête ; bagging, forêt, boosting et XGBoost sont à égalité compte
#   tenu de l'écart-type de validation croisée (≈ 0,006). XGBoost retrouve les
#   mêmes réglages que le gradient boosting (arbres de profondeur 2, 600
#   itérations, pas de 0,02) et les mêmes scores (log-loss 1,235, AUC 0,649),
#   en 5 fois moins de temps.
# - **Ridge, lasso, elastic net** : scores identiques (log-loss 1,247-1,248,
#   AUC 0,645-0,646). Le lasso (C = 0,1) ne garde que 41 coefficients sur 67 :
#   il écarte les indicatrices de non-réponse et une partie des modalités de
#   `SITUATION`, du lieu de vie, de la vie, de la situation et de la PCS des
#   parents, sans perte de performance. Ces modalités n'apportent donc rien de plus une fois
#   les autres connues, ce qui rejoint l'importance par permutation.
# - **Piste** : pour mieux repérer les précoces et les tardifs, on peut modifier
#   la règle de décision (seuils sur les probabilités, ou `class_weight` pour
#   rééquilibrer les classes), au prix d'une exactitude plus faible.
