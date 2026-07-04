import pandas as pd

try:
    from .clean_data import clean_data
except ImportError:
    from clean_data import clean_data


def read_data(data):
    data = pd.read_csv(data)
    return data


data = read_data(r'Module_1\src\data.csv')
data = clean_data(data)
data.to_csv(r'Module_1\src\cleaned_data.csv', index=False)
print(data.describe())
print(data.head())
print(data.info())
print(data.columns)
print(data.shape)
print(data.isnull().sum())
print(data.dtypes)
