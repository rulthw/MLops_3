import pickle
import pandas as pd
import os

df = pd.read_csv('~/MLops_3/datasets/data_test.csv')

X = df.drop(columns = ['LandAverageTemperature'])
y = df['LandAverageTemperature']

with open('/home/flow/MLops_3/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

score = model.score(X, y)
print("score=", score)
