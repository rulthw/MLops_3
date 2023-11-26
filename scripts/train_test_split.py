import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('~/MLops_3/datasets/data_processed.csv')

train, test = train_test_split(df, test_size= 0.3)

train.to_csv('~/MLops_3/datasets/data_train.csv', index=False)
test.to_csv('~/MLops_3/datasets/data_test.csv', index=False)
