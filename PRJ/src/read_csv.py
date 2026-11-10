"""Carga, limpieza y codificación del dataset de Melbourne (sin Pipeline).

Lo importan models/cross-validation.py, models/random-forest.py y src/predict_price.py.

    df = load_data()                    # datos limpios, categóricas como texto, numéricas con NaN
    df = load_data(imputar=True)        # ... y además sin huecos (ver impute_missing)
    X_rf  = encode(df, "rf")            # para Random Forest
    X_lin = encode(df, "linear")        # para regresión lineal
    X_cb  = df[FEATURES]                # CatBoost: tal cual, con cat_features=CAT_FEATURES
    y     = df[TARGET]

Al ejecutar este archivo se generan en datasets/:
    melbourne_limpio.csv    -> limpio y sin huecos, categóricas como texto (CatBoost)
    melbourne_numerico.csv  -> igual, pero categóricas en columnas 0/1 con get_dummies

Para codificar filas nuevas o modificadas (contrafactuales, casa nueva...) pasa siempre
ref=df, así se usan las mismas categorías, medianas y escalas que en el entrenamiento:
    encode(filas_nuevas, "rf", ref=df)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "Melbourne_housing_FULL.csv"
RESULTS_DIR = ROOT / "results"

# CSV generados por save_csvs()
CLEAN_CSV = ROOT / "datasets" / "melbourne_limpio.csv"        # categóricas como texto (CatBoost)
NUMERIC_CSV = ROOT / "datasets" / "melbourne_numerico.csv"    # categóricas con get_dummies

TARGET = "Price"
NUM_FEATURES = ["Rooms", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Distance"]

# Method (forma de venta) y SellerG (inmobiliaria) describen la VENTA, no la vivienda.
# Pon False si queréis responder 1.2 y 1.4 solo con características de la vivienda.
USAR_VARIABLES_VENTA = True

CAT_FEATURES = ["Type", "Regionname", "CouncilArea", "Suburb"]
if USAR_VARIABLES_VENTA:
    CAT_FEATURES += ["Method", "SellerG"]
FEATURES = NUM_FEATURES + CAT_FEATURES
# Address no se usa: es un identificador (casi una dirección distinta por fila) y no generaliza.

# get_dummies: categorías con menos ventas que esto se agrupan en "Otros"
# (evita cientos de columnas basadas en 2 o 3 casas).
MIN_VENTAS_CATEGORIA = 20

# Imputación: cada numérica se rellena con la mediana de viviendas parecidas.
# Se prueba el primer grupo; si ese grupo no tiene datos, el siguiente; al final, la mediana global.
GRUPOS_IMPUTACION = {
    "BuildingArea": [["Type", "Rooms"], ["Type"]],
    "Landsize":     [["Type", "Suburb"], ["Type", "Regionname"], ["Type"]],
    "YearBuilt":    [["Type", "Suburb"], ["Type", "Regionname"], ["Type"]],
    "Bathroom":     [["Type", "Rooms"], ["Rooms"]],
    "Car":          [["Type", "Rooms"], ["Rooms"]],
    "Rooms":        [["Type"]],
    "Distance":     [["Suburb"]],
}
ENTEROS = ["Rooms", "Bathroom", "Car", "YearBuilt"]

# Random Forest de scikit-learn >= 1.4 acepta NaN directamente
RF_ACEPTA_NAN = tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 4)


def load_data(path=DATA_PATH, imputar: bool = False) -> pd.DataFrame:
    """Devuelve las ventas con precio, con valores imposibles convertidos en NaN.

    imputar=True rellena además los huecos de las numéricas (ver impute_missing).
    """
    df = pd.read_csv(path).drop_duplicates()
    df = df.dropna(subset=[TARGET, "Distance"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Reglas de limpieza (valores imposibles o sin sentido -> NaN)
    df.loc[(df["BuildingArea"] <= 0) | (df["BuildingArea"] > 1000), "BuildingArea"] = np.nan
    df.loc[(df["Landsize"] <= 0) | (df["Landsize"] > 10_000), "Landsize"] = np.nan  # 0 = no informado
    df.loc[(df["YearBuilt"] < 1800) | (df["YearBuilt"] > 2018), "YearBuilt"] = np.nan
    df.loc[df["Bathroom"] > 8, "Bathroom"] = np.nan
    df.loc[df["Car"] > 10, "Car"] = np.nan

    df[NUM_FEATURES] = df[NUM_FEATURES].astype(float)
    for col in CAT_FEATURES:
        df[col] = df[col].fillna("Desconocido").astype(str)
    df = df.reset_index(drop=True)
    return impute_missing(df) if imputar else df


def impute_missing(df: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rellena los huecos de las numéricas con la mediana de viviendas parecidas.

    Ejemplo: a una casa de 3 habitaciones sin BuildingArea se le pone la mediana de las casas
    de 3 habitaciones, no la de todo el dataset (que mezcla pisos y mansiones).

    Por cada columna rellenada añade <columna>_imputado (1 = el valor era un hueco), para que
    el modelo sepa qué datos son reales y para poder filtrarlos (p. ej. la pregunta 1.4 entrena
    solo con viviendas con BuildingArea real: df[df["BuildingArea_imputado"] == 0]).

    No usa el precio, así que no filtra información del objetivo.
    """
    ref = df if ref is None else ref
    df = df.copy()
    for col, grupos in GRUPOS_IMPUTACION.items():
        if col not in df or not df[col].isna().any():
            continue
        df[f"{col}_imputado"] = df[col].isna().astype(int)
        for grupo in grupos:
            medianas = ref.groupby(grupo)[col].median().rename("_mediana")
            rellenos = df[grupo].join(medianas, on=grupo)["_mediana"]
            df[col] = df[col].fillna(rellenos)
        df[col] = df[col].fillna(ref[col].median())
        if col in ENTEROS:
            df[col] = df[col].round()
    return df


def dummies(df: pd.DataFrame, ref: pd.DataFrame | None = None, drop_first: bool = True) -> pd.DataFrame:
    """Categóricas -> columnas 0/1 con pd.get_dummies (p. ej. Type_t, Type_u).

    - Categorías con < MIN_VENTAS_CATEGORIA ventas en ref se agrupan en "Otros".
    - drop_first=True quita la primera categoría de cada columna (queda como referencia: si
      Type_t = 0 y Type_u = 0, es una casa). La regresión lineal lo necesita.
    - Las categorías salen de ref, así las filas nuevas tienen exactamente las mismas columnas.
    """
    ref = df if ref is None else ref
    partes = []
    for col in CAT_FEATURES:
        conteo = ref[col].value_counts()
        frecuentes = sorted(conteo[conteo >= MIN_VENTAS_CATEGORIA].index)
        categorias = frecuentes + (["Otros"] if len(frecuentes) < len(conteo) else [])
        valores = df[col].where(df[col].isin(frecuentes), "Otros")
        cat = pd.Categorical(valores, categories=categorias)
        partes.append(pd.get_dummies(cat, prefix=col, drop_first=drop_first, dtype=int).set_index(df.index))
    return pd.concat(partes, axis=1)


def encode(df: pd.DataFrame, model: str, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Convierte FEATURES en una tabla 100 % numérica para el modelo indicado.

    model="rf":
        - Categóricas -> un entero por categoría (orden alfabético). A un árbol le basta.
          Una categoría que no aparece en ref recibe -1.
        - Numéricas -> tal cual. Los NaN se dejan (el RF los gestiona) o, con scikit-learn
          antiguo, se rellenan con la mediana.
    model="linear":
        - Numéricas -> huecos rellenados (impute_missing) y estandarizadas (media 0, desv. 1),
          así los coeficientes son comparables entre sí (pregunta 1.2).
        - Categóricas -> get_dummies (ver dummies()).

    Nada de esto usa el precio, así que codificar antes de la validación cruzada no filtra
    información del test.
    """
    ref = df if ref is None else ref
    X = df[FEATURES].copy()

    if model == "rf":
        for col in CAT_FEATURES:
            X[col] = pd.Categorical(X[col], categories=sorted(ref[col].unique())).codes
        if not RF_ACEPTA_NAN:
            X[NUM_FEATURES] = X[NUM_FEATURES].fillna(ref[NUM_FEATURES].median())
        return X

    if model == "linear":
        num = impute_missing(X, ref=ref)[NUM_FEATURES]
        num_ref = impute_missing(ref[FEATURES])[NUM_FEATURES]
        num = (num - num_ref.mean()) / num_ref.std()
        return pd.concat([num, dummies(X, ref=ref)], axis=1)

    raise ValueError(f"Modelo no soportado: {model!r}. Usa 'rf' o 'linear' (CatBoost usa df[FEATURES]).")


def save_csvs(df: pd.DataFrame | None = None) -> None:
    """Guarda los datos limpios y sin huecos en dos versiones (mismas filas, mismo orden).

    melbourne_limpio.csv   -> numéricas + categóricas como texto + columnas *_imputado + Price.
    melbourne_numerico.csv -> lo mismo, pero las categóricas en columnas 0/1 (get_dummies).
    """
    df = load_data(imputar=True) if df is None else df
    if df[NUM_FEATURES].isna().any().any():
        df = impute_missing(df)

    marcas = [c for c in df.columns if c.endswith("_imputado")]
    limpio = df[NUM_FEATURES + marcas + CAT_FEATURES + [TARGET]]
    numerico = pd.concat([df[NUM_FEATURES + marcas], dummies(df), df[[TARGET]]], axis=1)

    limpio.to_csv(CLEAN_CSV, index=False)
    numerico.to_csv(NUMERIC_CSV, index=False)
    for ruta, tabla in [(CLEAN_CSV, limpio), (NUMERIC_CSV, numerico)]:
        print(f"Guardado {ruta.name}: {tabla.shape[0]:,} filas x {tabla.shape[1]} columnas, "
              f"{int(tabla.isna().sum().sum())} huecos")


if __name__ == "__main__":
    data = load_data()
    print("Filas con precio:", len(data))
    print("Huecos antes de imputar (%):")
    print((data[NUM_FEATURES].isna().mean() * 100).round(1).to_string())
    print(f"\nSi se borrasen las filas con algún hueco quedarían {len(data.dropna(subset=NUM_FEATURES)):,} "
          f"de {len(data):,}; por eso se imputan.")
    print("\nPrecio por tipo (AUD):")
    print(data.groupby("Type")[TARGET].describe()[["count", "50%", "mean"]].round(0).to_string())
    print()
    save_csvs(impute_missing(data))