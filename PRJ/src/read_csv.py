"""Carga y limpieza del dataset de Melbourne. Lo importan models/cross-validation.py y src/predict_price.py."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "Melbourne_housing_FULL.csv"
RESULTS_DIR = ROOT / "results"

TARGET = "Price"
NUM_FEATURES = ["Rooms", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Distance"]
CAT_FEATURES = ["Type", "Suburb", "Regionname"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Devuelve las ventas con precio, con valores imposibles convertidos en NaN.

    Los NaN se imputan después dentro de cada pipeline (solo con datos de entrenamiento).
    """
    df = pd.read_csv(path).drop_duplicates()
    df = df.dropna(subset=[TARGET, "Distance"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Reglas de limpieza (valores imposibles o sin sentido -> NaN)
    df.loc[(df["BuildingArea"] <= 0) | (df["BuildingArea"] > 1000), "BuildingArea"] = float("nan")
    df.loc[(df["Landsize"] <= 0) | (df["Landsize"] > 10_000), "Landsize"] = float("nan")  # 0 = no informado
    df.loc[(df["YearBuilt"] < 1800) | (df["YearBuilt"] > 2018), "YearBuilt"] = float("nan")
    df.loc[df["Bathroom"] > 8, "Bathroom"] = float("nan")
    df.loc[df["Car"] > 10, "Car"] = float("nan")

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Desconocido").astype(str)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    data = load_data()
    print("Filas con precio:", len(data))
    print("Nulos (%):")
    print((data[FEATURES].isna().mean() * 100).round(1).to_string())
    print("\nPrecio por tipo (AUD):")
    print(data.groupby("Type")[TARGET].describe()[["count", "50%", "mean"]].round(0).to_string())