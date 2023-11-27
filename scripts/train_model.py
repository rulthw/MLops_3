import xgboost as xgb
import pickle
import os
import pandas as pd

df = pd.read_csv("~/MLops_3/datasets/data_train.csv")

X = df.drop(columns = ["LandAverageTemperature"])
y = df["LandAverageTemperature"]

params = {
    "n_estimators":100,
    "max_depth": 4,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "colsample_bylevel": 0.8,
    "reg_lambda": 0.1,
    "random_state": 42,
}

model = xgb.XGBRegressor(**params)
model.fit(X, y)

with open("/home/flow/MLops_3/models/model.pkl", "wb") as f:
    pickle.dump(model, f)
