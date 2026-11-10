import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

# ---------- Load ----------
df = pd.read_csv("project-victor-asensio\datasets\Melbourne_housing_FULL.csv")
df = df.dropna(subset=["Price"])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ========== PART 1: What makes a house expensive? (all sales) ==========
features = ["Rooms", "Distance", "Bedroom2", "Bathroom", "Car",
            "Landsize", "BuildingArea", "YearBuilt"]
X = df[features].fillna(df[features].median())
y = df["Price"]

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf, X, y, cv=kf, scoring="r2")
print("All houses - random forest mean R²:", round(rf_scores.mean(), 3))

rf.fit(X, y)
importance = pd.Series(rf.feature_importances_, index=features).sort_values()

fig1, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(range(1, 6), rf_scores)
ax[0].set_title("All houses: random forest R² per fold")
ax[0].set_xlabel("Fold")
ax[0].set_ylabel("R²")
ax[1].barh(importance.index, importance.values)
ax[1].set_title("All houses: which features matter most")
ax[1].set_xlabel("Importance (sums to 1)")
plt.tight_layout()

# ========== PART 2: Predict price of land with no buildings ==========
land_features = ["Landsize", "Distance", "Propertycount", "Lattitude", "Longtitude"]
land = df[df["BuildingArea"].isna() & df["YearBuilt"].isna()].dropna(subset=land_features)
print("Vacant land rows:", len(land))

X_land = land[land_features]
y_land = land["Price"]

lr = LinearRegression()
rf_land = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

lr_scores = cross_val_score(lr, X_land, y_land, cv=kf, scoring="r2")
rf_land_scores = cross_val_score(rf_land, X_land, y_land, cv=kf, scoring="r2")
print("Land - linear regression mean R²:", round(lr_scores.mean(), 3))
print("Land - random forest mean R²:", round(rf_land_scores.mean(), 3))

pred_land = cross_val_predict(rf_land, X_land, y_land, cv=kf)
rf_land.fit(X_land, y_land)
lr.fit(X_land, y_land)

importance_land = pd.Series(rf_land.feature_importances_, index=land_features).sort_values()

fig2, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].bar(["Linear regression", "Random forest"],
          [lr_scores.mean(), rf_land_scores.mean()])
ax[0].set_title("Land: mean R² (higher is better)")
ax[0].set_ylabel("R²")

limit = y_land.quantile(0.99)  # zoom in on the cheapest 99% so the points are readable
ax[1].scatter(y_land, pred_land, alpha=0.3, s=10)
ax[1].plot([0, limit], [0, limit], "k--", label="Perfect prediction")
ax[1].set_xlim(0, limit)
ax[1].set_ylim(0, limit)
ax[1].set_title("Land: actual vs predicted price (random forest)")
ax[1].set_xlabel("Actual price (AUD)")
ax[1].set_ylabel("Predicted price (AUD)")
ax[1].legend()

ax[2].barh(importance_land.index, importance_land.values)
ax[2].set_title("Land: which features matter most")
ax[2].set_xlabel("Importance (sums to 1)")
plt.tight_layout()

# ---------- Predict the price of one lot ----------
new_lot = pd.DataFrame([{
    "Landsize": 600,               # m²
    "Distance": 12.0,              # km from the CBD
    "Propertycount": 5000,
    "Lattitude": -37.74,           # example coordinates
    "Longtitude": 144.96,
}])
print(f"Estimated land price (random forest): {rf_land.predict(new_lot)[0]:,.0f} AUD")

plt.show()