# write a program in python to demonstrate data free processing steps in an machine learning model. The steps will include handling machine data, encoding catagorical data,  feature scaling

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample data
data = {
    'Age': [25, np.nan, 35, 45, 50],
    'Salary': [50000, 60000, np.nan, 80000, 90000],
    'Country': ['France', 'Spain', 'Germany', 'Spain', 'Germany'],
    'Purchased': ['No', 'Yes', 'No', 'No', 'Yes']
}

# Create DataFrame
df = pd.DataFrame(data)

# Handling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# Encoding categorical data
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Country', 'Purchased'])
    ],
    remainder='passthrough'
)
df = np.array(ct.fit_transform(df))

# Feature scaling
scaler = StandardScaler()
df[:, -3:] = scaler.fit_transform(df[:, -3:])

print(df)