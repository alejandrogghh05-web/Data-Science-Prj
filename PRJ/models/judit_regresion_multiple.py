"""Parte de Judit: regresión lineal múltiple para casas MÁS GRANDES que las que ha visto el Random Forest.

Idea: un Random Forest no extrapola. Si una casa tiene más superficie construida que cualquiera de
las de entrenamiento, el bosque devuelve el precio de las casas más grandes que conoce (se "aplana").
Una regresión lineal sí sigue la pendiente superficie -> precio.

Qué hace este script
  1. Carga melbourne_limpio.csv y se queda con CASAS (Type == 'h') con BuildingArea real
     (BuildingArea_imputado == 0): la pendiente de superficie tiene que salir de datos reales.
  2. Prueba de extrapolación: entrena con las casas hasta el percentil P_ENTRENO de BuildingArea y
     evalúa en las más grandes. Compara Random Forest vs regresión múltiple (RMSE, MAE, R²).
  3. Modelo final para producción: entrena con todas las casas y define
     predecir_casa(...): usa el RF si la superficie está dentro del rango visto y la regresión si se sale.

Uso:  python judit_regresion_casas_grandes.py [ruta/melbourne_limpio.csv]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message="Found unknown categories")  # barrios raros -> categoría "infrecuente"

RANDOM_STATE = 42
P_ENTRENO = 0.80          # se entrena con el 80 % de casas más pequeñas y se evalúa en el 20 % más grande
ANIO_REF = 2018           # último año del dataset, para calcular la antigüedad

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "melbourne_limpio.csv"

NUM_RF = ["Rooms", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Distance"]
NUM_LIN = ["LogBuildingArea", "LogLandsize", "Rooms", "Bathroom", "Car", "Antiguedad", "Distance",
           "Landsize_imputado"]
CAT = ["Suburb"]


# ----------------------------------------------------------------------------- datos
def cargar_casas(csv=CSV):
    df = pd.read_csv(csv)
    casas = df[(df["Type"] == "h") & (df["BuildingArea_imputado"] == 0)].copy().reset_index(drop=True)
    return casas


def preparar(df):
    """Añade las columnas derivadas que usa la regresión (logs y antigüedad). No usa el precio."""
    df = df.copy()
    df["LogBuildingArea"] = np.log(df["BuildingArea"])
    df["LogLandsize"] = np.log(df["Landsize"])
    df["Antiguedad"] = ANIO_REF - df["YearBuilt"]
    return df


# ----------------------------------------------------------------------------- modelos
def modelo_rf():
    pre = ColumnTransformer([
        ("num", "passthrough", NUM_RF),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False), CAT),
    ])
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.33,
                               random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("prep", pre), ("model", rf)])


def modelo_lineal():
    """Regresión múltiple sobre log(Price). Numéricas estandarizadas -> coeficientes comparables;
    dummies de barrio sin escalar (si no, su coeficiente pierde interpretación)."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_LIN),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                              drop="first", sparse_output=False), CAT),
    ])
    return Pipeline([("prep", pre), ("model", LinearRegression())])


def ajustar(train):
    train = preparar(train)
    rf = modelo_rf().fit(train[NUM_RF + CAT], train["Price"])
    lin = modelo_lineal().fit(train[NUM_LIN + CAT], np.log(train["Price"]))
    return rf, lin


def pred_rf(rf, df):
    return rf.predict(preparar(df)[NUM_RF + CAT])


def pred_lin(lin, df):
    # el modelo se ajusta sobre log(Price): se deshace el log. Se corrige el sesgo de la
    # retransformación (smearing de Duan) con el factor guardado en lin.smearing_.
    return np.exp(lin.predict(preparar(df)[NUM_LIN + CAT])) * getattr(lin, "smearing_", 1.0)


def ajustar_con_smearing(train):
    rf, lin = ajustar(train)
    t = preparar(train)
    residuos = np.log(t["Price"]) - lin.predict(t[NUM_LIN + CAT])
    lin.smearing_ = float(np.mean(np.exp(residuos)))
    return rf, lin


def metricas(y, p):
    return {"RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p)}


# ----------------------------------------------------------------------------- 2. prueba de extrapolación
def prueba_extrapolacion(casas):
    corte = casas["BuildingArea"].quantile(P_ENTRENO)
    train, test = casas[casas["BuildingArea"] <= corte], casas[casas["BuildingArea"] > corte]
    print(f"Casas con superficie real: {len(casas)}")
    print(f"Entreno: {len(train)} casas (<= {corte:.0f} m²) | Test (más grandes): {len(test)} casas "
          f"({test['BuildingArea'].min():.0f}-{test['BuildingArea'].max():.0f} m²)\n")

    rf, lin = ajustar_con_smearing(train)
    y = test["Price"]
    p_rf, p_lin = pred_rf(rf, test), pred_lin(lin, test)
    # modelo combinado: RF dentro del rango visto, regresión fuera (aquí todo el test está fuera)
    p_mix = np.where(test["BuildingArea"] <= train["BuildingArea"].max(), p_rf, p_lin)

    tabla = pd.DataFrame({"Random Forest": metricas(y, p_rf),
                          "Regresión múltiple": metricas(y, p_lin),
                          "Combinado": metricas(y, p_mix)}).T
    print("=== Casas MÁS GRANDES que las de entrenamiento (AUD) ===")
    print(tabla.round(3).to_string())

    sesgo = pd.DataFrame({"sesgo medio RF": [(p_rf - y).mean()], "sesgo medio Regresión": [(p_lin - y).mean()]})
    print("\nSesgo medio (predicho - real, AUD); negativo = infravalora:")
    print(sesgo.round(0).to_string(index=False))

    # por tramos de superficie: el RF se aplana, la regresión sigue la pendiente
    tramos = pd.qcut(test["BuildingArea"], 4, duplicates="drop")
    por_tramo = pd.DataFrame({"real": y, "RF": p_rf, "Regresión": p_lin}).groupby(tramos, observed=True).mean()
    print("\nPrecio medio por tramo de superficie (m²):")
    print(por_tramo.round(0).to_string())

    # control: en el rango ya visto, ¿no pierde la regresión frente al RF? (validación simple 80/20 aleatoria)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(train))
    corte_idx = int(0.8 * len(train))
    tr, va = train.iloc[idx[:corte_idx]], train.iloc[idx[corte_idx:]]
    rf2, lin2 = ajustar_con_smearing(tr)
    ctrl = pd.DataFrame({"Random Forest": metricas(va["Price"], pred_rf(rf2, va)),
                         "Regresión múltiple": metricas(va["Price"], pred_lin(lin2, va))}).T
    print("\n=== Control: casas del rango ya visto (validación 80/20) ===")
    print(ctrl.round(3).to_string())
    return tabla


# ----------------------------------------------------------------------------- 3. modelo final
class ModeloCasaGrande:
    """RF dentro del rango de superficie visto, regresión múltiple fuera de él."""

    def __init__(self, casas):
        self.rf, self.lin = ajustar_con_smearing(casas)
        self.max_area = casas["BuildingArea"].max()
        self.max_area_p95 = casas["BuildingArea"].quantile(0.95)

    def predecir(self, filas, umbral=None):
        """filas: DataFrame con Rooms, Bathroom, Car, Landsize, BuildingArea, YearBuilt, Distance,
        Landsize_imputado (0/1) y Suburb. umbral: superficie a partir de la cual se usa la regresión
        (por defecto, el percentil 95 de entrenamiento, donde el RF ya tiene pocas casas)."""
        umbral = self.max_area_p95 if umbral is None else umbral
        p_rf, p_lin = pred_rf(self.rf, filas), pred_lin(self.lin, filas)
        usa_lin = filas["BuildingArea"].to_numpy() > umbral
        return pd.Series(np.where(usa_lin, p_lin, p_rf), index=filas.index), pd.Series(
            np.where(usa_lin, "regresión", "random forest"), index=filas.index)

    def coeficientes(self):
        nombres = self.lin.named_steps["prep"].get_feature_names_out()
        coef = pd.Series(self.lin.named_steps["model"].coef_, index=nombres)
        return coef[coef.index.str.startswith("num__")].sort_values(key=abs, ascending=False)


def demo(casas):
    mod = ModeloCasaGrande(casas)
    print("\n=== Coeficientes de la regresión (log precio, variables estandarizadas) ===")
    print(mod.coeficientes().round(3).to_string())

    base = casas.iloc[[0]].copy()
    filas = pd.concat([base.assign(BuildingArea=a) for a in (150, 300, 500, 800)], ignore_index=True)
    filas["Landsize"] = np.maximum(filas["Landsize"], filas["BuildingArea"] * 1.5)
    pred, quien = mod.predecir(filas)
    print(f"\nEjemplo: misma casa en {base['Suburb'].iloc[0]} aumentando la superficie construida "
          f"(el RF ha visto casas hasta {mod.max_area:.0f} m²)")
    print(pd.DataFrame({"BuildingArea": filas["BuildingArea"], "Precio_estimado": pred.round(0),
                        "modelo_usado": quien}).to_string(index=False))


if __name__ == "__main__":
    casas = cargar_casas()
    prueba_extrapolacion(casas)
    demo(casas)
