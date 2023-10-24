# %% Imports

import pandas as pd
from mafese import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
from pyfume.Clustering import Clusterer
from pyfume.EstimateAntecendentSet import AntecedentEstimator
from pyfume.EstimateConsequentParameters import ConsequentEstimator
from pyfume.SimpfulModelBuilder import SugenoFISBuilder
from pyfume.Tester import SugenoFISTester
from mafese import UnsupervisedSelector, FilterSelector, LassoSelector, TreeSelector
from mafese import SequentialSelector, RecursiveSelector, MhaSelector, MultiMhaSelector
import numpy as np
from numpy import clip, column_stack, argmax
import tensorflow as tf

# %% Open the data as a Dataframe
flight_Data = pd.read_csv('train.csv')

# %% Exclude some "a priori" features and get dummies with others

flight_Data = pd.get_dummies(flight_Data, columns=['CARRIER_NAME'])
flight_Data =  pd.get_dummies(flight_Data, columns=['DEPARTING_AIRPORT'])

flight_Data = flight_Data.drop(['DEP_TIME_BLK', 'PREVIOUS_AIRPORT'], axis = 1)
train_data = flight_Data

# %%
headers = train_data.columns
headers = headers.tolist()
del headers[2]

# %% Random sampling 

delayed_data = train_data[train_data['DEP_DEL15'] == 1]
delayed_data = delayed_data.sample(n=5000, random_state=1213)
ontime_data = train_data[train_data['DEP_DEL15'] == 0]
ontime_data = ontime_data.sample(n=len(delayed_data), random_state=783)

train_data = pd.concat([ontime_data, delayed_data], axis=0, ignore_index=True)

# %%  Pre processing of data (Removing duplicate data, NDA, etc...)

flight_Data = train_data.dropna()
flight_Data = train_data.drop_duplicates()

# %% Data Understanding (1)

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 6]
# Scatter plots
train_data.plot(kind="scatter", x="AWND", y="DEP_DEL15")


# %% Data Understading (2)

## Import pearsonr function from scipy -- calculate the correlation and p-value between two columns
from scipy.stats import pearsonr
import seaborn as sns

## Calculate the correlations between the columns
df_corrs = train_data.corr(method='spearman')

## Calculate the p-value, as the second element of the response from the pearsonr function. 
pval = train_data.corr(method=lambda x, y: pearsonr(x, y)[1])

## Establish the mask, to hide values without a given statistical significance
ptg_stat_sig = 0.1/100
mask = pval > ptg_stat_sig

## Plot the correlation matrix using seaborn's heatmap function
plt.subplots(figsize=(139, 139))
heatmap = sns.heatmap(df_corrs, mask = mask, square = True, cmap = 'coolwarm', annot = True)


# %% Data Understanding (3)
colors = ['blue' if d == 1 else 'red' for d in train_data['DEP_DEL15']]

plt.scatter(train_data['DAY_OF_WEEK'], train_data['PRCP'], c=colors , s=5)

# Add a legend
plt.scatter([], [], c='blue', label='Delayed', s=10)
plt.scatter([], [], c='red', label='Not Delayed', s=10)
plt.legend()

# Add axis labels and title
plt.xlabel('Day of the Week')
plt.ylabel('Volume of Precipitation')
plt.grid(True)

# Show the plot
plt.show()

# %%  Renaming the collumn names in the dataset

flight_Data.columns = [f'V{i}' for i in range(1, len(flight_Data.columns) + 1)]
column_names = flight_Data.columns.tolist()
column_names = [item for item in column_names if item != 'V3']  

# %%  Normalizing the data

X = flight_Data.drop(columns=['V3']) 
y = flight_Data['V3'] 
y = y.values
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# %%  Feature Selection

X_scaled = X_scaled.values
data = Data(X_scaled, y)
data.split_train_test(test_size=0.2, inplace=True)

# Feature selector with a Tree Structure
#feat_selector = TreeSelector(problem="classification", estimator="tree")

# Feature selector witha  Filter Method
feat_selector = FilterSelector(problem='classification', method='SPEARMAN', n_features = 10)

#feat_selector = SequentialSelector(problem="classification", estimator="rf", n_features=20, direction="forward")
#feat_selector = RecursiveSelector(problem="classification", estimator="rf", n_features=30)

# Feature selector with a Metaheuristic Algorithm
#feat_selector = MhaSelector(problem="classification", estimator="knn", optimizer="OriginalACOR", optimizer_paras=None, transfer_func="vstf_01", obj_name="AS")
#feat_selector.fit(data.X_train, data.y_train)

feat_selector.fit(data.X_train, data.y_train)
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)
selected_indices = np.where(feat_selector.selected_feature_solution == 1)[0]

X =X_scaled[:, selected_indices]

# %% PCA Understanding

from sklearn.decomposition import PCA

# Initialize PCA with the desired number of components
n_components = 20  # Adjust this based on the desired number of components
pca = PCA(n_components=n_components)

# Fit PCA to the standardized data and transform it
X_pca = pca.fit_transform(X_scaled)

# Access the principal components and loadings
principal_components = pca.components_
explained_variance_ratio = pca.explained_variance_ratio_

# Assess variable influence by examining loadings in the first principal component
first_principal_component_loading = principal_components[0]


top_10_indices = np.argsort(-np.abs(first_principal_component_loading))[:4]
top_10_loadings = first_principal_component_loading[top_10_indices]

selected_headers = [headers[i] for i in top_10_indices]


# Print the loadings for the first principal component
print("Loadings for the first principal component:")
print(first_principal_component_loading)

# Assess variable influence by examining explained variance ratios
print("Explained Variance Ratios:")
print(explained_variance_ratio)

# %% Data Manipulation

X = pd.DataFrame(X)
X_names = [headers[idx] for idx in selected_indices]
y = pd.DataFrame(y)
y_names = 'DEP_DEL15'
train_data = pd.concat([X, y], axis=1, ignore_index=True)

train_data.columns = X_names + [y_names]

# %% Data Correlation


## Import pearsonr function from scipy -- calculate the correlation and p-value between two columns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

## Calculate the correlations between the columns
df_corrs = train_data.corr(method='spearman')

## Calculate the p-value, as the second element of the response from the pearsonr function. 
pval = train_data.corr(method=lambda x, y: pearsonr(x, y)[1])

## Establish the mask, to hide values without a given statistical significance
ptg_stat_sig = 0.1/100
mask = pval > ptg_stat_sig

## Plot the correlation matrix using seaborn's heatmap function
plt.subplots(figsize=(10, 10))
heatmap = sns.heatmap(df_corrs, mask = mask, square = True, cmap = 'coolwarm', annot = True)