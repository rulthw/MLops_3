import xgboost as xgb
import pickle
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model_training")

df = pd.read_csv('~/MLops_3/datasets/data_train.csv')

X = df.drop(columns = ['LandAverageTemperature'])
y = df['LandAverageTemperature']

with mlflow.start_run():

    params = {
    "n_estimators":100,
    "max_depth": 2,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "colsample_bylevel": 0.8,
    "reg_lambda": 0.1,
    "random_state": 42}

    reg = xgb.XGBRegressor(**params)

    mlflow.sklearn.log_model(reg,
                             artifact_path="model",
                             registered_model_name="model")
    mlflow.log_artifact(local_path="/home/flow/MLops_3/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.log_param("n_estimators", params["n_estimators"])
    mlflow.log_param("max_depth", params["max_depth"])
    mlflow.log_param("learning_rate", params["learning_rate"])
    mlflow.log_param("subsample", params["subsample"])
    mlflow.log_param("colsample_bytree", params["colsample_bytree"])
    mlflow.log_param("colsample_bylevel", params["colsample_bylevel"])
    mlflow.log_param("reg_lambda", params["reg_lambda"])
    mlflow.log_param("random_state", params["random_state"])
    mlflow.end_run()

reg.fit(X, y)

with open('/home/flow/MLops_3/models/model.pkl', 'wb') as f:
    pickle.dump(reg, f)
