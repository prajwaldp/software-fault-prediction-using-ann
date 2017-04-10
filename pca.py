"""
Principle Component Analysis:
Performs Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


sns.set(color_codes=True)

# Read the dataset into a pandas DataFrame
df = pd.read_csv('camel-1.6.csv',
                 usecols=[3, 4, 5, 6, 7, 8, 23],
                 dtype={'wmc': np.float32, 'dit': np.float32,
                        'noc': np.float32, 'cbo': np.float32,
                        'rfc': np.float32, 'lcom': np.float32},
                 converters={23: lambda x: 1 if x > '0' else 0})

X = df.loc[:, ('wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom')].values
target = df['bug'].values

# Preprocessing data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


pca = PCA(n_components=2, svd_solver='randomized')
reduced_x_pca = pca.fit_transform(X)


# Building a scatterplot
for i in range(2):
    data = reduced_x_pca[0:-1][target == i]
    sns.regplot(x='First Principle Component', y='Second Principle Component', data=data)

plt.show()
