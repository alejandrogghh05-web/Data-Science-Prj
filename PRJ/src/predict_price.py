"""Preguntas 1.3 y 1.4 del README.

1.3  Qué tipo de vivienda (h casa, t townhouse, u unidad) da más "beneficio".
     Indicador proxy: precio por m² construido (el dataset no tiene costes).
       - Bruto: mediana de Price / BuildingArea por tipo.
       - Controlado: precio por m² predicho por cada modelo al cambiar solo el tipo,
         manteniendo fijos superficie construida, barrio, distancia, año y habitaciones.
1.4  Rentabilidad de comprar un terreno y construir (el dataset no contiene terrenos sin construir).
     Modelos entrenados solo con casas con superficie construida. Con ellos se estima
     (a) el valor por m² construido que paga el mercado y (b) el precio del terreno =
     precio de la casa - valor de lo construido; se compara con el coste de construcción SUPUESTO.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling import build_models  # noqa: E402
from read_csv import CAT_FEATURES, NUM_FEATURES, RESULTS_DIR, TARGET, load_data  # noqa: E402

# 1. Load the data and keep rows with a price
df = pd.read_csv("PRJ\datasets\Melbourne_housing_FULL.csv")
df = df.dropna(subset=["Price"])


RESULTS_DIR.mkdir(exist_ok=True)

# ---------------- SUPUESTOS DE 1.4 (no salen del dataset; cámbialos aquí) ----------------
COSTE_CONSTRUCCION_M2 = 2_000   # AUD por m² construido
OTROS_COSTES_PCT = 0.15         # permisos, tasas, imprevistos y gastos de venta, sobre el coste de obra
SENSIBILIDAD_COSTE_M2 = [1_500, 2_000, 2_500, 3_000, 3_500]

df = load_data()
ref = df[df["BuildingArea"].notna()].copy()
print(f"Ventas con superficie construida: {len(ref)}")

# ======================= 1.3 Tipo de vivienda =======================
# Se excluye Landsize: el terreno propio es parte de lo que distingue una casa de una unidad.
num_13 = [c for c in NUM_FEATURES if c != "Landsize"]
cols_13 = num_13 + CAT_FEATURES
ref["PrecioM2"] = ref[TARGET] / ref["BuildingArea"]

res_13 = ref.groupby("Type").agg(n=(TARGET, "size"), precio_mediano=(TARGET, "median"),
                                  m2_construidos_mediano=("BuildingArea", "median"),
                                  bruto_precio_m2=("PrecioM2", "median"))
models_13 = {name: m.fit(ref[cols_13], ref[TARGET]) for name, m in build_models(num_13, CAT_FEATURES).items()}

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
models_14 = {name: m.fit(casas[cols_14], casas[TARGET]) for name, m in build_models(NUM_FEATURES, cat_14).items()}
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
        b = valor_por_m2(model, g)
        pred = model.predict(g[cols_14])
        terreno = np.median(pred - b * g["BuildingArea"])          # precio estimado del terreno
        mercado = g[TARGET].median()                                # precio de mercado de la zona (casas)
        area = g["BuildingArea"].median()
        coste = area * COSTE_CONSTRUCCION_M2 * (1 + OTROS_COSTES_PCT)
        margen = mercado - terreno - coste
        filas.append({
            "Modelo": name, "Zona": region, "n": len(g), "valor_m2_construido": b,
            "coste_max_m2_para_rentable": b / (1 + OTROS_COSTES_PCT),
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
    b = valor_por_m2(model, casas)
    terreno = np.median(model.predict(casas[cols_14]) - b * casas["BuildingArea"])
    mercado, area = casas[TARGET].median(), casas["BuildingArea"].median()
    sens[name] = {c: 100 * (mercado - terreno - area * c * (1 + OTROS_COSTES_PCT))
                     / (terreno + area * c * (1 + OTROS_COSTES_PCT)) for c in SENSIBILIDAD_COSTE_M2}
print("\nMargen % sobre inversión (todas las zonas) según coste de construcción por m²:")
print(pd.DataFrame(sens).rename_axis("AUD/m²").round(1).to_string())


def precio_terreno(modelo, b, suburb, regionname, distancia, landsize, area_ref=None):
    """Estima el precio de un terreno: precio de una casa de referencia en esa parcela menos el valor de lo construido."""
    zona = casas[casas["Suburb"] == suburb]
    base = (zona if len(zona) else casas)[NUM_FEATURES].median()
    base["Landsize"], base["Distance"] = landsize, distancia
    if area_ref is not None:
        base["BuildingArea"] = area_ref
    x = pd.DataFrame([{**base.to_dict(), "Suburb": suburb, "Regionname": regionname}])[cols_14]
    return modelo.predict(x)[0] - b * base["BuildingArea"]


ejemplo = casas.loc[casas["Suburb"] == casas["Suburb"].value_counts().index[0]].iloc[0]
print(f"\nEjemplo: parcela de 600 m² en {ejemplo['Suburb']} ({ejemplo['Distance']} km del centro)")
for name, model in models_14.items():
    p = precio_terreno(model, valor_por_m2(model, casas), ejemplo["Suburb"], ejemplo["Regionname"], ejemplo["Distance"], 600)
    print(f"  {name}: {p:,.0f} AUD")