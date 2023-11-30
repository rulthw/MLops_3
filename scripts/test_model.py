import pickle
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model_testing")

df = pd.read_csv('~/MLops_3/datasets/data_test.csv')

X = df.drop(columns = ['LandAverageTemperature'])
y = df['LandAverageTemperature']

with open('/home/flow/MLops_3/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with mlflow.start_run():
    predict = model.predict(X)
    rmse = mean_squared_error(y, predict, squared=False)
    mae = mean_absolute_error(y, predict)
    r2 = r2_score(y, predict)
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    print("RMSE =", rmse)
    print("MAE =", mae)
    print("R2 =", r2)
    mlflow.end_run()
