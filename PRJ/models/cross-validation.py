"""Preguntas 1.1 y 1.2 del README.

1.1  Compara regresión lineal, Random Forest y CatBoost con k-fold repetido (RMSE, MAE, R²)
     y contrasta si las diferencias son significativas (t de remuestreo corregida, Bonferroni).
1.2  Importancia de variables en los tres modelos y coincidencia entre ellos.
"""


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedKFold, cross_validate
from itertools import combinations
from src.modeling import build_models, original_variable  # noqa: E402
from src.read_csv import FEATURES, RESULTS_DIR, TARGET, load_data  # noqa: E402

N_SPLITS, N_REPEATS = 5, 2
RESULTS_DIR.mkdir(exist_ok=True)

df = load_data()
X, y = df[FEATURES], df[TARGET]
print(f"Ventas usadas: {len(df)}")

# ======================= 1.1 Validación cruzada =======================
cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)  # mismas particiones para los 3 modelos
scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}

folds, summary = {}, []
for name, model in build_models().items():
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

# ======================= 1.2 Variables que encarecen =======================
models = {name: model.fit(X, y) for name, model in build_models().items()}

lin = models["Regresión lineal"]
names = lin.named_steps["prep"].get_feature_names_out()
groups = pd.Series([original_variable(n) for n in names], index=names)

coef = pd.Series(lin.named_steps["model"].coef_, index=names)  # AUD por desviación típica de la variable
imp_lin = coef.abs().groupby(groups).sum()

rf = models["Random Forest"]
imp_rf = pd.Series(rf.named_steps["model"].feature_importances_, index=names).groupby(groups).sum()

imp_cb = pd.Series(models["CatBoost"].feature_importance())

print("\n=== 1.2 Coeficientes estandarizados de la regresión lineal (AUD por +1 desv. típica) ===")
signed = coef[[n for n in names if n.startswith("num__") and "missingindicator" not in n or n.startswith("cat__Type_")]]
print(signed.sort_values(key=abs, ascending=False).round(0).to_string())

table = pd.DataFrame({"Regresión lineal": imp_lin, "Random Forest": imp_rf, "CatBoost": imp_cb})
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