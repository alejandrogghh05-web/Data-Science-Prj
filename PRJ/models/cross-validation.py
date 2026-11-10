import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------- 1. Vacant land sales ----------
df = pd.read_csv("project-victor-asensio\datasets\Melbourne_housing_FULL.csv")
df = df.dropna(subset=["Price"]).copy()
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

land = df[df["BuildingArea"].isna() & df["YearBuilt"].isna()].copy()
FEATURES = ["Landsize", "Distance", "Lattitude", "Longtitude"]
TARGET = "Price"
land = land.dropna(subset=FEATURES + [TARGET])
land = land[land["Landsize"] > 0].sort_values("Date").reset_index(drop=True)
print("Vacant land sales:", len(land))

# ---------- 2. Split by date ----------
VALIDATION_START = "2017-07-01"
TEST_START = "2018-01-01"

train = land[land["Date"] < VALIDATION_START]
validation = land[(land["Date"] >= VALIDATION_START) & (land["Date"] < TEST_START)]
test = land[land["Date"] >= TEST_START]
print(f"Train: {len(train)}  Validation: {len(validation)}  Test: {len(test)}")

def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE_AUD": mean_absolute_error(y_true, y_pred),
        "RMSE_AUD": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

# ---------- 3. Tune the random forest on train only ----------
time_series_cv = TimeSeriesSplit(n_splits=5)
scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}

rf_grid = GridSearchCV(
    RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    param_grid={"max_depth": [4, 6, 8, None], "min_samples_leaf": [3, 5, 10]},
    cv=time_series_cv,
    scoring="neg_root_mean_squared_error",
)
rf_grid.fit(train[FEATURES], train[TARGET])
print("Best random forest settings:", rf_grid.best_params_)

model_candidates = {
    "Linear regression": LinearRegression(),
    "Random Forest (tuned)": rf_grid.best_estimator_,
}

# ---------- 4. Cross-validation on train, then check on validation ----------
cv_rows, validation_rows = [], []
for model_name, estimator in model_candidates.items():
    cv_scores = cross_validate(
        estimator, train[FEATURES], train[TARGET],
        cv=time_series_cv, scoring=scoring, n_jobs=1, error_score="raise",
    )
    cv_rows.append({
        "Model": model_name,
        "CV_R2_mean": cv_scores["test_r2"].mean(),
        "CV_R2_std": cv_scores["test_r2"].std(),
        "CV_MAE_mean": -cv_scores["test_mae"].mean(),
        "CV_RMSE_mean": -cv_scores["test_rmse"].mean(),
    })

    estimator.fit(train[FEATURES], train[TARGET])
    val_pred = estimator.predict(validation[FEATURES])
    validation_rows.append({"Model": model_name, **regression_metrics(validation[TARGET], val_pred)})

cv_table = pd.DataFrame(cv_rows).set_index("Model")
validation_table = pd.DataFrame(validation_rows).set_index("Model")
display(cv_table.round(2))
display(validation_table.round(2))

# Choose the model using validation only
selected_model_name = validation_table["RMSE_AUD"].idxmin()
print("Selected model:", selected_model_name)

# ---------- 5. Final fit on train + validation, one look at the test set ----------
final_model = model_candidates[selected_model_name]
development = pd.concat([train, validation], ignore_index=True)
final_model.fit(development[FEATURES], development[TARGET])

test_predictions = final_model.predict(test[FEATURES])
test_metrics = regression_metrics(test[TARGET], test_predictions)
display(pd.DataFrame([test_metrics], index=[f"Final test - {selected_model_name}"]).round(2))

fig, ax = plt.subplots(figsize=(7, 6))
limit = test[TARGET].quantile(0.99)
ax.scatter(test[TARGET], test_predictions, alpha=0.4, s=12)
ax.plot([0, limit], [0, limit], "k--", label="Perfect prediction")
ax.set(title=f"Test 2018+ ({selected_model_name})",
       xlabel="Actual price (AUD)", ylabel="Predicted price (AUD)",
       xlim=(0, limit), ylim=(0, limit))
ax.legend()
plt.show()

def distance_km(lat, lon):
    """Distance in km from the Melbourne CBD to a point (haversine formula)."""
    R = 6371  # Earth's radius in km
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(CBD_LAT), np.radians(CBD_LON)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------- 6. Estimate a plot from coordinates ----------
def predict_plot_price(model, landsize_m2, lat, lon):
    distance = distance_km(lat, lon)  # function from the earlier code
    plot = pd.DataFrame([{
        "Landsize": landsize_m2,
        "Distance": distance,
        "Lattitude": lat,
        "Longtitude": lon,
    }])
    return model.predict(plot[FEATURES])[0], distance

price, d = predict_plot_price(final_model, 600, lat=-37.74, lon=144.96)
print(f"Estimated plot price: {price:,.0f} AUD ({d:.1f} km from the CBD)")