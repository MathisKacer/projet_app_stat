# projet_app_stat

## `data/`

Chargement du jeu de données ESCAPAD.

- Source : bucket S3 (MinIO) d'Onyxia, non versionné dans le dépôt :
  `s3://mathiskacer2/diffusion/projet_escapad_marylou_mathis/ESCAP.csv`
- `load_data.py` : fonction `load_escap_data()` qui télécharge le CSV
  depuis S3 (identifiants pris automatiquement dans l'environnement
  Onyxia) et le charge dans un DataFrame pandas, sans mise en cache
  locale (rechargement depuis S3 à chaque appel).
  - Le CSV source est séparé par `;` ; les valeurs manquantes (cellules
    vides ou espace) sont converties en `NaN`.

- `labels.py` : libellés des variables et des modalités (d'après le sujet).
- `preprocess.py` : fonction `preprocess()` qui enchaîne :
  1. suppression des âges au premier alcool > 25 ans et discrétisation de
     `Q19A` par quantiles pondérés dans `Q19A_CLASSE` (NA → « Non concerné »,
     jamais bu) ; `Q19A` continue est conservée ;
  2. complétion de `Q04` à partir de `Q04A` / `Q04B`, modalité « Non concerné »
     pour les questions filtrées, variable de synthèse `SITUATION` ;
  3. libellés des modalités et NA → « Non répondu » pour les autres variables.- `features.py` : variables explicatives retenues pour les modèles et leur
  encodage (`make_preprocessor()`) : scores ordinaux + indicatrice de
  non-réponse pour les variables ordinales, one-hot pour les nominales.

Utilisation (dans un notebook, par exemple) :

```python
from data import load_escap_data, preprocess

df = preprocess(load_escap_data())
```

Dépendances : `pandas`, `s3fs` (voir `requirements.txt`).

## `notebooks/`

Notebooks interactifs au format « percent » (cellules délimitées par `# %%`),
à exécuter cellule par cellule dans VS Code (Python Interactive) ou avec
Jupytext.

- `01_visualisation.py` : statistique descriptive pondérée par `pm17B` :
  valeurs manquantes, distribution de la cible `Q19A` et valeurs aberrantes,
  distribution des prédicteurs, âge moyen au premier alcool par modalité,
  profil des non-réponses à `Q19A`, V de Cramér entre prédicteurs.
- `02_preprocessing.py` : chaque étape du prétraitement avec ses statistiques
  de contrôle (histogramme + boîte à moustaches pondérée de `Q19A`, classes
  obtenues, complétion de `Q04`, tableau de `SITUATION`, non-réponses).
- `03_modeles.py` : prédiction de `Q19A` (jeunes ayant déjà bu) : moyenne,
  Ridge, arbre, bagging, forêt aléatoire, gradient boosting (classique et par
  histogrammes), stacking. Hyperparamètres par validation croisée pondérée,
  comparaison sur un échantillon test, importance des variables par permutation.
