"""Variables explicatives et encodage pour les modèles de prédiction de `Q19A`.

S'applique au jeu issu de `preprocess` (variables catégorielles libellées,
non-réponses en modalité « Non répondu »).

- Variables ordinales (ou binaires) : score entier 1..k selon l'ordre des
  modalités ; « Non répondu » est imputé par la médiane et signalé par une
  indicatrice, pour garder à la fois l'ordre et l'information de non-réponse.
- Variables nominales : indicatrices (one-hot) ; les modalités de moins de
  `MIN_FREQUENCY` individus sont regroupées dans une modalité « rare ».
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from .labels import NON_REPONDU

# `SITUATION` résume `Q04`, `Q04A` et `Q04B`, qui ne sont donc pas repris.
ORDINAL = ["Q03", "Q05", "Q06A", "Q06B", "B08A", "B08B"]
NOMINAL = ["SITUATION", "Q08", "Q08C", "Q09A1", "Q09B1", "Q10A1", "Q10B1"]
FEATURES = ORDINAL + NOMINAL

MIN_FREQUENCY = 30


def ordinal_scores(X: pd.DataFrame) -> np.ndarray:
    """Rang de la modalité (1..k), NaN pour « Non répondu »."""
    return np.column_stack(
        [
            np.where(X[c].astype(str) == NON_REPONDU, np.nan, X[c].cat.codes + 1.0)
            for c in X.columns
        ]
    )


def _ordinal_names(transformer, input_features):
    return np.asarray(input_features, dtype=object)


def make_preprocessor() -> ColumnTransformer:
    """Transforme les variables explicatives en matrice numérique."""
    ordinal = make_pipeline(
        FunctionTransformer(ordinal_scores, feature_names_out=_ordinal_names),
        SimpleImputer(strategy="median", add_indicator=True),
    )
    nominal = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=MIN_FREQUENCY,
        sparse_output=False,
    )
    return ColumnTransformer(
        [("ord", ordinal, ORDINAL), ("nom", nominal, NOMINAL)],
        verbose_feature_names_out=False,
    )
