"""Preguntas 1.1, 1.2, 1.3 y 1.4 del README.

1.1  Compara regresión lineal, Random Forest y CatBoost con k-fold repetido (RMSE, MAE, R²)
     y contrasta si las diferencias son significativas (t de remuestreo corregida, Bonferroni).
1.2  Importancia de variables en los cuatro modelos y coincidencia entre ellos.

1.3  Qué tipo de vivienda (h casa, t townhouse, u unidad) da más "beneficio".
     Indicador proxy: precio por m² construido (el dataset no tiene costes).
       - Bruto: mediana de Price / BuildingArea por tipo.
       - Controlado: precio por m² predicho por cada modelo al cambiar solo el tipo,
         manteniendo fijos superficie construida, barrio, distancia, año y habitaciones.
1.4  Rentabilidad de comprar un terreno y construir (el dataset no contiene terrenos sin construir).
     Modelos entrenados solo con casas con superficie construida: precio del terreno =
     precio de la casa - valor de lo construido; se compara con el coste de construcción SUPUESTO.

Cuarto enfoque (teoría S04): regresión lineal regularizada con LassoCV (alpha elegido por
validación cruzada interna). Además se aplica lo visto en S04:
  - elección justificada del nº de folds (media y dispersión según K),
  - relación filas/columnas para valorar si la regularización puede aportar,
  - control con 5 columnas de ruido: Lasso debe ponerlas a cero exacto.
"""


import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # carpeta src/ junto a models/
from modeling import RANDOM_STATE, _preprocessor, build_models, original_variable  # noqa: E402
from read_csv import CAT_FEATURES, FEATURES, NUM_FEATURES, RESULTS_DIR, TARGET, load_data  # noqa: E402

N_SPLITS, N_REPEATS = 5, 2
RESULTS_DIR.mkdir(exist_ok=True)
LASSO = "Lasso (LassoCV)"
ALPHAS = np.logspace(-2, 6, 25)  # Price está en AUD: los coeficientes son grandes y alpha debe llegar lejos


def build_all(num_cols=NUM_FEATURES, cat_cols=CAT_FEATURES):
    """Los tres modelos de clase + regresión lineal regularizada (Lasso, variables estandarizadas)."""
    models = build_models(num_cols, cat_cols)
    models[LASSO] = Pipeline([
        ("prep", _preprocessor(num_cols, cat_cols, scale=True)),  # escalar es obligatorio al regularizar
        ("model", LassoCV(alphas=ALPHAS, cv=5, max_iter=20_000, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    return models


df = load_data()
X, y = df[FEATURES], df[TARGET]
print(f"Ventas usadas: {len(df)}")

# ======================= 1.1 Validación cruzada =======================
cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)  # mismas particiones para los 3 modelos
scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}

folds, summary = {}, []
for name, model in build_all().items():
    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
    folds[name] = {"RMSE": -res["test_RMSE"], "MAE": -res["test_MAE"], "R2": res["test_R2"]}
    summary.append({
        "Modelo": name,
        "RMSE": folds[name]["RMSE"].mean(), "RMSE_std": folds[name]["RMSE"].std(ddof=1),
        "MAE": folds[name]["MAE"].mean(), "MAE_std": folds[name]["MAE"].std(ddof=1),
        "R2": folds[name]["R2"].mean(), "R2_std": folds[name]["R2"].std(ddof=1),
    })
summary = pd.DataFrame(summary).set_index("Modelo")
print("\n=== 1.1 Métricas (media y desviación entre particiones; RMSE y MAE en AUD) ===")
print(summary.round(3).to_string())
summary.to_csv(RESULTS_DIR / "1_1_metricas_cv.csv")


def corrected_resampled_ttest(a, b, n_train, n_test):
    """t de remuestreo corregida (Nadeau y Bengio; Bouckaert y Frank para k-fold repetido)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)


n_test = len(X) // N_SPLITS
n_train = len(X) - n_test
pairs = list(combinations(folds, 2))
tests = []
for a, b in pairs:
    diff = folds[a]["RMSE"] - folds[b]["RMSE"]
    t, p = corrected_resampled_ttest(folds[a]["RMSE"], folds[b]["RMSE"], n_train, n_test)
    tests.append({
        "Comparación": f"{a} vs {b}", "Dif. RMSE media (AUD)": diff.mean(), "t": t, "p": p,
        "p (Bonferroni)": min(1.0, p * len(pairs)),
    })
tests = pd.DataFrame(tests).set_index("Comparación")
tests["¿Significativa (5 %)?"] = tests["p (Bonferroni)"] < 0.05
print("\n=== 1.1 Contraste de diferencias en RMSE (t corregida, 10 particiones) ===")
print(tests.round(4).to_string())
tests.to_csv(RESULTS_DIR / "1_1_contrastes.csv")

# ---- Elección del nº de folds (S04): media y dispersión del RMSE según K, regresión lineal ----
lin_cv = build_models()["Regresión lineal"]
filas_k = []
for k in (3, 5, 10, 20):
    t0 = time.time()
    r = -cross_validate(lin_cv, X, y, cv=KFold(k, shuffle=True, random_state=42),
                        scoring="neg_root_mean_squared_error", n_jobs=1)["test_score"]
    filas_k.append({"K": k, "RMSE medio": r.mean(), "Desv. típica": r.std(ddof=1),
                    "Desv./media (%)": 100 * r.std(ddof=1) / r.mean(), "Tiempo (s)": time.time() - t0})
tabla_k = pd.DataFrame(filas_k).set_index("K")
print("\n=== 1.1 Elección de K (regresión lineal): estabilidad frente a coste ===")
print(tabla_k.round(2).to_string())
tabla_k.to_csv(RESULTS_DIR / "1_1_eleccion_k.csv")

# ======================= 1.2 Variables que encarecen =======================
models = {name: model.fit(X, y) for name, model in build_all().items()}

lin = models["Regresión lineal"]
names = lin.named_steps["prep"].get_feature_names_out()
groups = pd.Series([original_variable(n) for n in names], index=names)

coef = pd.Series(lin.named_steps["model"].coef_, index=names)  # AUD por desviación típica de la variable
imp_lin = coef.abs().groupby(groups).sum()

rf = models["Random Forest"]
imp_rf = pd.Series(rf.named_steps["model"].feature_importances_, index=names).groupby(groups).sum()

imp_cb = pd.Series(models["CatBoost"].feature_importance())

las = models[LASSO]
coef_las = pd.Series(las.named_steps["model"].coef_, index=names)
imp_las = coef_las.abs().groupby(groups).sum()
print(f"\nLasso: alpha elegido = {las.named_steps['model'].alpha_:,.2f}; "
      f"coeficientes a cero exacto: {(coef_las == 0).sum()} de {len(coef_las)}")
print("Variables originales que Lasso elimina por completo:",
      sorted(set(groups) - set(groups[coef_las != 0])) or "ninguna")

# ---- Relación filas/columnas (S04): ¿puede la regularización aportar algo? ----
n_cols = len(names)
print(f"Relación filas/columnas tras codificar: {len(X) / n_cols:.0f} a 1 ({len(X)} filas, {n_cols} columnas)")

# ---- Control con ruido (S04): 5 columnas aleatorias que Lasso debe descartar ----
rng = np.random.RandomState(42)
X_prep = lin.named_steps["prep"].transform(X)
X_ruido = np.hstack([X_prep, rng.rand(X_prep.shape[0], 5)])
lasso_ruido = LassoCV(alphas=ALPHAS, cv=5, max_iter=20_000, random_state=RANDOM_STATE, n_jobs=-1).fit(X_ruido, y)
coef_ruido = lasso_ruido.coef_[-5:]
print(f"Control de ruido: coeficientes de las 5 columnas aleatorias en Lasso = {coef_ruido.round(2).tolist()} "
      f"-> ¿todas a cero? {bool((coef_ruido == 0).all())}")

print("\n=== 1.2 Coeficientes estandarizados de la regresión lineal (AUD por +1 desv. típica) ===")
signed = coef[[n for n in names if n.startswith("num__") and "missingindicator" not in n or n.startswith("cat__Type_")]]
print(signed.sort_values(key=abs, ascending=False).round(0).to_string())

table = pd.DataFrame({"Regresión lineal": imp_lin, "Random Forest": imp_rf, "CatBoost": imp_cb, LASSO: imp_las})
table = table.reindex(FEATURES).fillna(0)
table = table / table.sum() * 100  # cada modelo suma 100 %
top3 = {c: set(table[c].nlargest(3).index) for c in table}
table["En top-3 de (nº modelos)"] = [sum(v in s for s in top3.values()) for v in table.index]
table = table.sort_values(["En top-3 de (nº modelos)", "Random Forest"], ascending=False)
print("\n=== 1.2 Importancia por variable (% sobre el total de cada modelo) ===")
print(table.round(1).to_string())
table.to_csv(RESULTS_DIR / "1_2_importancias.csv")

rank_corr = table[list(top3)].corr(method="spearman")
print("\nCorrelación de rangos de Spearman entre modelos:")
print(rank_corr.round(2).to_string())
print("Variables en el top-3 de los tres modelos:", sorted(set.intersection(*top3.values())) or "ninguna")


# ======================= 1.3 Tipo de vivienda =======================
# ---------------- SUPUESTOS DE 1.4 (no salen del dataset; cámbialos aquí) ----------------
COSTE_CONSTRUCCION_M2 = 2_000   # AUD por m² construido
OTROS_COSTES_PCT = 0.15         # permisos, tasas, imprevistos y gastos de venta, sobre el coste de obra
SENSIBILIDAD_COSTE_M2 = [1_500, 2_000, 2_500, 3_000, 3_500]

ref = df[df["BuildingArea"].notna()].copy()
print(f"\nVentas con superficie construida: {len(ref)}")

# Se excluye Landsize: el terreno propio es parte de lo que distingue una casa de una unidad.
num_13 = [c for c in NUM_FEATURES if c != "Landsize"]
cols_13 = num_13 + CAT_FEATURES
ref["PrecioM2"] = ref[TARGET] / ref["BuildingArea"]

res_13 = ref.groupby("Type").agg(n=(TARGET, "size"), precio_mediano=(TARGET, "median"),
                                  m2_construidos_mediano=("BuildingArea", "median"),
                                  bruto_precio_m2=("PrecioM2", "median"))
models_13 = {name: m.fit(ref[cols_13], ref[TARGET]) for name, m in build_all(num_13, CAT_FEATURES).items()}

for name, model in models_13.items():
    vals = {}
    for tipo in res_13.index:
        X_t = ref[cols_13].copy()
        X_t["Type"] = tipo  # mismo inmueble, solo cambia el tipo
        vals[tipo] = np.median(model.predict(X_t) / ref["BuildingArea"])
    res_13[f"controlado_{name}"] = pd.Series(vals)

print("\n=== 1.3 Precio por m² construido (AUD/m²) por tipo ===")
print(res_13.round(0).to_string())
print("Tipo con mayor precio/m²:", {c: res_13[c].idxmax() for c in res_13.columns if "precio_m2" in c or "controlado" in c})
res_13.to_csv(RESULTS_DIR / "1_3_precio_m2_por_tipo.csv")

# ======================= 1.4 Terreno sin construir =======================
casas = ref[ref["Type"] == "h"].copy()
cat_14 = [c for c in CAT_FEATURES if c != "Type"]
cols_14 = NUM_FEATURES + cat_14
models_14 = {name: m.fit(casas[cols_14], casas[TARGET]) for name, m in build_all(NUM_FEATURES, cat_14).items()}
print(f"\nCasas con superficie construida usadas en 1.4: {len(casas)}")


def valor_por_m2(model, data):
    """Precio que el modelo asigna a cada m² construido adicional (pendiente local, +-20 % de superficie)."""
    lo, hi = data[cols_14].copy(), data[cols_14].copy()
    lo["BuildingArea"] *= 0.8
    hi["BuildingArea"] *= 1.2
    return (model.predict(hi).mean() - model.predict(lo).mean()) / (0.4 * data["BuildingArea"].mean())


filas = []
for name, model in models_14.items():
    for region, g in [("TODAS", casas)] + list(casas.groupby("Regionname")):
        b_m2 = valor_por_m2(model, g)
        pred = model.predict(g[cols_14])
        terreno = np.median(pred - b_m2 * g["BuildingArea"])        # precio estimado del terreno
        mercado = g[TARGET].median()                                # precio de mercado de la zona (casas)
        area = g["BuildingArea"].median()
        coste = area * COSTE_CONSTRUCCION_M2 * (1 + OTROS_COSTES_PCT)
        margen = mercado - terreno - coste
        filas.append({
            "Modelo": name, "Zona": region, "n": len(g), "valor_m2_construido": b_m2,
            "coste_max_m2_para_rentable": b_m2 / (1 + OTROS_COSTES_PCT),
            "precio_terreno_estimado": terreno, "precio_mercado_casa": mercado,
            "terreno_%_del_precio": 100 * terreno / mercado, "coste_construccion_total": coste,
            "margen_AUD": margen, "margen_%_sobre_inversion": 100 * margen / (terreno + coste),
        })
res_14 = pd.DataFrame(filas).set_index(["Modelo", "Zona"])
print(f"\n=== 1.4 Supuestos: coste {COSTE_CONSTRUCCION_M2} AUD/m² + {OTROS_COSTES_PCT:.0%} de otros costes ===")
print(res_14.round(0).to_string())
res_14.to_csv(RESULTS_DIR / "1_4_terreno_por_zona.csv")

sens = {}
for name, model in models_14.items():
    b_m2 = valor_por_m2(model, casas)
    terreno = np.median(model.predict(casas[cols_14]) - b_m2 * casas["BuildingArea"])
    mercado, area = casas[TARGET].median(), casas["BuildingArea"].median()
    sens[name] = {c: 100 * (mercado - terreno - area * c * (1 + OTROS_COSTES_PCT))
                     / (terreno + area * c * (1 + OTROS_COSTES_PCT)) for c in SENSIBILIDAD_COSTE_M2}
print("\nMargen % sobre inversión (todas las zonas) según coste de construcción por m²:")
print(pd.DataFrame(sens).rename_axis("AUD/m²").round(1).to_string())


def precio_terreno(modelo, b_m2, suburb, regionname, distancia, landsize, area_ref=None):
    """Estima el precio de un terreno: precio de una casa de referencia en esa parcela menos el valor de lo construido."""
    zona = casas[casas["Suburb"] == suburb]
    base = (zona if len(zona) else casas)[NUM_FEATURES].median()
    base["Landsize"], base["Distance"] = landsize, distancia
    if area_ref is not None:
        base["BuildingArea"] = area_ref
    x = pd.DataFrame([{**base.to_dict(), "Suburb": suburb, "Regionname": regionname}])[cols_14]
    return modelo.predict(x)[0] - b_m2 * base["BuildingArea"]


ejemplo = casas.loc[casas["Suburb"] == casas["Suburb"].value_counts().index[0]].iloc[0]
print(f"\nEjemplo: parcela de 600 m² en {ejemplo['Suburb']} ({ejemplo['Distance']} km del centro)")
for name, model in models_14.items():
    p_t = precio_terreno(model, valor_por_m2(model, casas), ejemplo["Suburb"], ejemplo["Regionname"], ejemplo["Distance"], 600)
    print(f"  {name}: {p_t:,.0f} AUD")