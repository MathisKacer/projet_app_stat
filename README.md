# projet_app_stat

## `data/`

Chargement du jeu de données ESCAPAD.

- Source : bucket S3 (MinIO) d'Onyxia, non versionné dans le dépôt :
  `s3://mathiskacer2/diffusion/projet_escapad_marylou_mathis/ESCAP.csv`
- `load_data.py` : fonction `load_escap_data()` qui télécharge le CSV
  depuis S3 (identifiants pris automatiquement dans l'environnement
  Onyxia) et met en cache une copie locale dans `data/raw/` (ignoré par
  git).
  - `load_escap_data(use_cache=True)` (défaut) : réutilise le cache local
    s'il existe.
  - `load_escap_data(refresh_cache=True)` : force le retéléchargement.
  - Le CSV source est séparé par `;` ; les valeurs manquantes (cellules
    vides ou espace) sont converties en `NaN`.

Utilisation :

```python
from data import load_escap_data

df = load_escap_data()
```

Dépendances : `pandas`, `s3fs` (voir `requirements.txt`).
