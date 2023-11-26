import pandas as pd

df = pd.read_csv('~/MLops_3/datasets/data.csv')

# Delete Uncertainty Data
df = df.drop(columns = ['LandAverageTemperatureUncertainty', 
                        'LandMaxTemperatureUncertainty', 
                        'LandMinTemperatureUncertainty', 
                        'LandAndOceanAverageTemperatureUncertainty'])

# Fill NaN values
df["LandAverageTemperature"].fillna(df["LandAverageTemperature"].mean(), inplace = True)
df["LandMaxTemperature"].fillna(df["LandMaxTemperature"].mean(), inplace = True)
df["LandMinTemperature"].fillna(df["LandMinTemperature"].mean(), inplace = True)
df["LandAndOceanAverageTemperature"].fillna(df["LandAndOceanAverageTemperature"].mean(), inplace = True)

# Convert Fahrenheit to Celsius
df["LandAverageTemperature"] = df["LandAverageTemperature"].apply(lambda x: (x * 1.8) + 32)
df["LandMaxTemperature"] = df["LandMaxTemperature"].apply(lambda x: (x * 1.8) + 32)
df["LandMinTemperature"] = df["LandMinTemperature"].apply(lambda x: (x * 1.8) + 32)
df["LandAndOceanAverageTemperature"] = df["LandAndOceanAverageTemperature"].apply(lambda x: (x * 1.8) + 32)

# Get Year and Month
df["dt"] = pd.to_datetime(df["dt"])
df["Year"] = df["dt"].dt.year
df["Month"] = df["dt"].dt.month
df = df.drop(columns = ["dt"])

df.to_csv('~/MLops_3/datasets/data_processed.csv', index = False)
