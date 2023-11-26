#!/usr/bin/python3
import pandas as pd

datafile = 'https://raw.githubusercontent.com/amankharwal/Website-data/master/GlobalTemperatures.csv'

df= pd.read_csv(datafile)

df.to_csv("~/MLops_3/datasets/data.csv", columns=df.columns)
