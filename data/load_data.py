"""Chargement du jeu de données ESCAPAD depuis le stockage S3 Onyxia.

Le fichier source vit sur le bucket S3 du projet et n'est jamais versionné
dans le dépôt git : ce module le télécharge (avec mise en cache locale
optionnelle) et le charge dans un DataFrame pandas correctement typé.

Utilisation :
    from data import load_escap_data
    df = load_escap_data()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import s3fs

S3_PATH = "mathiskacer2/diffusion/projet_escapad_marylou_mathis/ESCAP.csv"
LOCAL_CACHE_PATH = Path(__file__).parent / "raw" / "ESCAP.csv"

# Dans les données brutes, les valeurs manquantes sont codées par des
# cellules vides ou ne contenant que des espaces.
NA_VALUES = ["", " "]


def get_file_system() -> s3fs.S3FileSystem:
    """Retourne un système de fichiers S3 basé sur les identifiants Onyxia.

    S3FileSystem lit automatiquement les identifiants (clé, secret, jeton
    de session) depuis les variables d'environnement standard AWS_* déjà
    injectées dans les services Onyxia ; seul l'endpoint doit être précisé.
    """
    return s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})


def load_escap_data(use_cache: bool = True, refresh_cache: bool = False) -> pd.DataFrame:
    """Charge le jeu de données ESCAPAD.

    Parameters
    ----------
    use_cache : bool
        Si True (défaut), réutilise une copie locale déjà téléchargée
        (data/raw/ESCAP.csv) plutôt que de retélécharger depuis S3.
    refresh_cache : bool
        Si True, force le retéléchargement depuis S3 même si une copie
        locale existe, et la met à jour.

    Returns
    -------
    pd.DataFrame
        Les données ESCAPAD, avec les valeurs manquantes codées en NaN.
    """
    if use_cache and not refresh_cache and LOCAL_CACHE_PATH.exists():
        return pd.read_csv(LOCAL_CACHE_PATH, sep=";", na_values=NA_VALUES)

    fs = get_file_system()
    with fs.open(S3_PATH, "r") as f:
        df = pd.read_csv(f, sep=";", na_values=NA_VALUES)

    if use_cache:
        LOCAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(LOCAL_CACHE_PATH, sep=";", index=False)

    return df


if __name__ == "__main__":
    data = load_escap_data()
    print(f"{data.shape[0]} lignes, {data.shape[1]} colonnes")
    print(data.head())
