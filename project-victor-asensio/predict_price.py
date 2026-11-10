import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

# 1. Load the data and keep rows with a price
df = pd.read_csv(r"D:\BigDataProjects\project1-datascience\datasets\Melbourne_housing_FULL.csv")
df = df.dropna(subset=["Price"])

# 2. Keep land rows (no building data) that have size, distance and coordinates
land = df[df["BuildingArea"].isna() & df["YearBuilt"].isna()].copy()
land = land.dropna(subset=["Landsize", "Distance", "Lattitude", "Longtitude"])
print("Land rows:", len(land))

# 3. Coordinates of the Melbourne CBD, used to compute distance from coordinates
CBD_LAT, CBD_LON = -37.8136, 144.9631

def distance_km(lat, lon):
    """Distance in km from the Melbourne CBD to a point (haversine formula)."""
    R = 6371  # Earth's radius in km
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(CBD_LAT), np.radians(CBD_LON)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# 4. Features and target
features = ["Landsize", "Distance"]
X = land[features]
y = land["Price"]

# 5. Cross-validation: compare random forest and linear regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
lr = LinearRegression()

print("Random forest mean R²:", round(cross_val_score(rf, X, y, cv=kf, scoring="r2").mean(), 3))
print("Linear regression mean R²:", round(cross_val_score(lr, X, y, cv=kf, scoring="r2").mean(), 3))

# 6. Fit the final model on all land rows
rf.fit(X, y)

# 7. Predict the price of a plot, from coordinates or from distance
def predict_land(landsize_m2, lat=None, lon=None, distance=None):
    if distance is None:
        distance = distance_km(lat, lon)
    new_plot = pd.DataFrame([{"Landsize": landsize_m2, "Distance": distance}])
    return rf.predict(new_plot)[0], distance

# Example 1: you have coordinates
price, d = predict_land(600, lat=-37.74, lon=144.96)
print(f"Coordinates -> {d:.1f} km from the CBD, estimated price: {price:,.0f} AUD")

# Example 2: you only know the distance
price, d = predict_land(600, distance=12)
print(f"Distance 12 km, estimated price: {price:,.0f} AUD")