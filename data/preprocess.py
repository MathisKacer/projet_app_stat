"""Prétraitement du jeu de données ESCAPAD.

Étapes (voir `preprocess`) :

1. cible `Q19A` : suppression des âges > 25 ans (la JDC se fait au plus tard à
   25 ans) et discrétisation par quantiles pondérés dans `Q19A_CLASSE`, les
   NA devenant « Non concerné » (jeunes n'ayant jamais bu) ;
2. situation : `Q04` complété à partir de `Q04A` / `Q04B`, modalité
   « Non concerné » pour les questions filtrées, et variable de synthèse
   `SITUATION` ;
3. autres prédicteurs : libellés des modalités et NA en « Non répondu ».

Les variables catégorielles sont renvoyées en `pd.Categorical` ordonnées
(les codes d'origine restent accessibles via `.cat.codes`).

Utilisation :
    from data import load_escap_data, preprocess

    df = preprocess(load_escap_data())
"""

import numpy as np
import pandas as pd

from .labels import NON_CONCERNE, NON_REPONDU, PREDICTORS, TARGET, VARIABLES, WEIGHT

AGE_MAX = 25
TARGET_CLASS = "Q19A_CLASSE"
SITUATION = "SITUATION"
SITUATION_VARS = ("Q04", "Q04A", "Q04B")


def to_labelled(codes: pd.Series, var: str, extra=(NON_REPONDU,)) -> pd.Series:
    """Remplace les codes par leurs libellés ; les NA deviennent `extra[-1]`."""
    labels = VARIABLES[var][1]
    categories = list(labels.values()) + [c for c in extra if c not in labels.values()]
    values = codes.map(labels).fillna(extra[-1])
    return pd.Series(pd.Categorical(values, categories=categories, ordered=True), index=codes.index)


# --- 1. Cible ---------------------------------------------------------------


def clean_target(df: pd.DataFrame, age_max: int = AGE_MAX) -> pd.DataFrame:
    """Supprime les individus dont l'âge au premier alcool dépasse `age_max`."""
    return df[~(df[TARGET] > age_max)].copy()


def target_thresholds(df: pd.DataFrame, n_bins: int = 4) -> list[int]:
    """Seuils de discrétisation de `Q19A` à partir des quantiles pondérés.

    L'âge étant entier, on retient pour chaque quantile l'âge dont la fonction
    de répartition pondérée en est la plus proche : la classe k regroupe les
    âges dans ]seuil_{k-1}, seuil_k].
    """
    obs = df[TARGET].notna()
    dist = df.loc[obs, WEIGHT].groupby(df.loc[obs, TARGET]).sum().sort_index()
    cdf = dist.cumsum() / dist.sum()
    thresholds = set()
    for q in np.arange(1, n_bins) / n_bins:
        age = (cdf - q).abs().idxmin()
        if age < cdf.index[-1]:
            thresholds.add(int(age))
    return sorted(thresholds)


def _class_labels(thresholds: list[int]) -> list[str]:
    labels = [f"≤ {thresholds[0]} ans"]
    for low, high in zip(thresholds, thresholds[1:]):
        labels.append(f"{high} ans" if high == low + 1 else f"{low + 1}-{high} ans")
    labels.append(f"≥ {thresholds[-1] + 1} ans")
    return labels


def discretize_target(df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """Ajoute `Q19A_CLASSE` : classes de quantiles, NA -> « Non concerné »."""
    df = df.copy()
    thresholds = target_thresholds(df, n_bins)
    labels = _class_labels(thresholds)
    classes = pd.cut(df[TARGET], bins=[-np.inf, *thresholds, np.inf], labels=labels)
    df[TARGET_CLASS] = pd.Categorical(
        classes.astype("object").fillna(NON_CONCERNE),
        categories=labels + [NON_CONCERNE],
        ordered=True,
    )
    return df


# --- 2. Situation -----------------------------------------------------------


def complete_q04(df: pd.DataFrame) -> pd.DataFrame:
    """Déduit `Q04` quand il manque et qu'une seule question filtrée est renseignée."""
    df = df.copy()
    q04_na, a, b = df["Q04"].isna(), df["Q04A"].notna(), df["Q04B"].notna()
    df.loc[q04_na & a & ~b, "Q04"] = 1
    df.loc[q04_na & b & ~a, "Q04"] = 2
    return df


def build_situation(df: pd.DataFrame) -> pd.DataFrame:
    """Recode `Q04`, `Q04A`, `Q04B` et ajoute la variable de synthèse `SITUATION`.

    `Q04` fait foi : `Q04A` (resp. `Q04B`) vaut « Non concerné » pour ceux qui
    ont arrêté (resp. suivent) leurs études, même s'ils y ont répondu, et
    « Non répondu » pour une non-réponse alors qu'ils étaient concernés.
    """
    df = complete_q04(df)
    q04 = df["Q04"]
    extra = (NON_CONCERNE, NON_REPONDU)
    q04a = to_labelled(df["Q04A"], "Q04A", extra)
    q04b = to_labelled(df["Q04B"], "Q04B", extra)
    q04a[q04 == 2] = NON_CONCERNE
    q04b[q04 == 1] = NON_CONCERNE

    situation = pd.Series(NON_REPONDU, index=df.index, dtype="object")
    etudes, arret = q04 == 1, q04 == 2
    situation[etudes] = "Études : " + q04a[etudes].astype(str)
    situation[arret] = "Arrêt : " + q04b[arret].astype(str)
    order = [f"Études : {m}" for m in q04a.cat.categories if m != NON_CONCERNE]
    order += [f"Arrêt : {m}" for m in q04b.cat.categories if m != NON_CONCERNE]

    df["Q04"] = to_labelled(q04, "Q04")
    df["Q04A"], df["Q04B"] = q04a, q04b
    df[SITUATION] = pd.Categorical(situation, categories=order + [NON_REPONDU], ordered=True)
    return df


# --- 3. Autres prédicteurs --------------------------------------------------


def encode_missing(df: pd.DataFrame, variables=None) -> pd.DataFrame:
    """Libellés des modalités, NA -> « Non répondu »."""
    df = df.copy()
    if variables is None:
        variables = [v for v in PREDICTORS if v not in SITUATION_VARS]
    for var in variables:
        df[var] = to_labelled(df[var], var)
    return df


def preprocess(df: pd.DataFrame, n_bins: int = 4, age_max: int = AGE_MAX) -> pd.DataFrame:
    """Enchaîne les trois étapes de prétraitement."""
    df = clean_target(df, age_max)
    df = discretize_target(df, n_bins)
    df = build_situation(df)
    return encode_missing(df)
