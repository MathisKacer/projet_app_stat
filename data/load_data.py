"""Chargement du jeu de données ESCAPAD depuis le stockage S3 Onyxia.

Utilisation :
    from data import load_escap_data

    df = load_escap_data()
"""

import pandas as pd
import s3fs

S3_PATH = "mathiskacer2/diffusion/projet_escapad_marylou_mathis/ESCAP.csv"
S3_ENDPOINT = "https://minio.lab.sspcloud.fr"

# Dans les données brutes, les valeurs manquantes sont codées par des
# cellules vides ou ne contenant que des espaces.
NA_VALUES = ("", " ")

# Les décimales (poids de sondage pm17B) utilisent une virgule.
DECIMAL = ","


def load_escap_data() -> pd.DataFrame:
    """Télécharge et charge le jeu de données ESCAPAD depuis S3."""
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT})
    with fs.open(S3_PATH, "r") as f:
        return pd.read_csv(f, sep=";", na_values=NA_VALUES, decimal=DECIMAL)


if __name__ == "__main__":
    df = load_escap_data()
    print(df.shape)
    print(df.head())
