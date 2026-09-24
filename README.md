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

Utilisation (dans un notebook, par exemple) :

```python
from data import load_escap_data

df = load_escap_data()
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
