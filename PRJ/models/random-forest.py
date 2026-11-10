import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from scipy import stats

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Configuración ----------------
BASE = Path(__file__).resolve().parent.parent      # .../PRJ
RUTA_CSV = BASE / "datasets" / "Melbourne_housing_FULL.csv"
RANDOM_STATE = 42
N_SPLITS, N_REPEATS = 5, 3
RF_PARAMS = dict(n_estimators=300, max_features=0.4, min_samples_leaf=2,
                 n_jobs=-1, random_state=RANDOM_STATE)

FIG_DIR = BASE / "figuras_rf"; FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = BASE / "resultados_rf"; OUT_DIR.mkdir(exist_ok=True)

# ---------------- Estilo de los gráficos ----------------
AZUL, NARANJA, AGUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
COLOR_TIPO = {"h": AZUL, "u": NARANJA, "t": AGUA}
NOMBRE_TIPO = {"h": "Casa (h)", "u": "Unidad / piso (u)", "t": "Townhouse (t)"}

sns.set_theme(style="whitegrid", font_scale=0.95, rc={
    "axes.edgecolor": "#d6d5d0", "grid.color": "#ebeae6", "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "figure.dpi": 110})

fmt_aud = mtick.FuncFormatter(lambda v, _: f"{v/1e6:.1f} M" if abs(v) >= 1e6 else f"{v/1e3:.0f} k")

def guardar(fig, nombre):
    fig.savefig(FIG_DIR / f"{nombre}.png", dpi=150, bbox_inches="tight")

##
    
df_raw = pd.read_csv(RUTA_CSV)
print(f"Dataset original: {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas")
df_raw.head()

##

nulos = (df_raw.isna().mean() * 100).sort_values()
nulos = nulos[nulos > 0]

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.barh(nulos.index, nulos.values, color=AZUL, height=0.6)
for i, v in enumerate(nulos.values):
    ax.text(v + 0.8, i, f"{v:.0f} %", va="center", fontsize=9, color="#52514e")
ax.set_xlim(0, nulos.max() * 1.15)
ax.set_xlabel("% de valores nulos")
ax.set_title("Valores nulos por columna")
ax.grid(axis="y", visible=False)
guardar(fig, "00_nulos"); plt.show()

## 

NUMERICAS = ["Rooms", "Bathroom", "Car", "Distance", "Landsize", "BuildingArea", "YearBuilt",
             "Lattitude", "Longtitude", "Propertycount", "sale_year", "sale_month"]
CAT_BAJA = ["Type", "Regionname", "CouncilArea"]   # pocas categorías -> OrdinalEncoder
CAT_ALTA = ["Suburb"]                               # ~350 barrios -> TargetEncoder
FEATURES = NUMERICAS + CAT_BAJA + CAT_ALTA


def limpiar(df):
    df = df.copy()
    for c in ["Price", "Rooms", "Bathroom", "Car", "Distance", "Landsize", "BuildingArea",
              "YearBuilt", "Lattitude", "Longtitude", "Propertycount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Price"]).reset_index(drop=True)

    fecha = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["sale_year"], df["sale_month"] = fecha.dt.year, fecha.dt.month

    df.loc[df["BuildingArea"] <= 10, "BuildingArea"] = np.nan
    df.loc[(df["YearBuilt"] < 1830) | (df["YearBuilt"] > df["sale_year"]), "YearBuilt"] = np.nan
    df.loc[(df["Landsize"] == 0) & (df["Type"] == "h"), "Landsize"] = np.nan
    for c in ["Landsize", "BuildingArea"]:
        df.loc[df[c] > df[c].quantile(0.999), c] = np.nan
    df[NUMERICAS] = df[NUMERICAS].astype(float)   # PDP de sklearn no admite columnas enteras
    return df


df = limpiar(df_raw)
X, y = df[FEATURES], df["Price"]
df[FEATURES + ["Price"]].to_csv(OUT_DIR / "datos_limpios.csv", index=False)
print(f"Tras limpiar: {len(df):,} viviendas con precio")
print(df["Type"].value_counts().rename(NOMBRE_TIPO))

##

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(y, bins=80, color=AZUL)
axes[0].xaxis.set_major_formatter(fmt_aud)
axes[0].set_title("Precio (AUD): muy asimétrico")
axes[0].set_xlabel("Precio")
axes[1].hist(np.log10(y), bins=80, color=AZUL)
axes[1].set_title("log10(precio): casi simétrico")
axes[1].set_xlabel("log10(precio)")
for ax in axes:
    ax.set_ylabel("Viviendas"); ax.grid(axis="x", visible=False)
fig.tight_layout() 
guardar(fig, "01_distribucion_precio") 
plt.show()