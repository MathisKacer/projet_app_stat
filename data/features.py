"""Variables explicatives et encodage pour les modèles de prédiction de `Q19A`.

S'applique au jeu issu de `preprocess` (variables catégorielles libellées,
non-réponses en modalité « Non répondu »).

Toutes les variables sont encodées en indicatrices (one-hot), y compris les
variables ordinales : la non-réponse reste une modalité à part entière
(« Non répondu »), sans imputation, comme dans le reste du projet. Les
modalités de moins de `MIN_FREQUENCY` individus sont regroupées dans une
modalité « rare ».
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# `SITUATION` résume `Q04`, `Q04A` et `Q04B`, qui ne sont donc pas repris.
ORDINAL = ["Q03", "Q05", "Q06A", "Q06B", "B08A", "B08B"]
NOMINAL = ["SITUATION", "Q08", "Q08C", "Q09A1", "Q09B1", "Q10A1", "Q10B1"]
FEATURES = ORDINAL + NOMINAL

MIN_FREQUENCY = 30


def make_preprocessor() -> ColumnTransformer:
    """Transforme les variables explicatives en matrice d'indicatrices."""
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_FREQUENCY,
                    sparse_output=False,
                ),
                FEATURES,
            )
        ],
        verbose_feature_names_out=False,
    )
