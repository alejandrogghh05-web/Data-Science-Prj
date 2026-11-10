import pandas as pd

df = pd.read_csv("project-victor-asensio\datasets\Melbourne_housing_FULL.csv")
df = df.dropna(subset=["Price"])

df.info()