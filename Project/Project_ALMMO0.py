# %% Imports

from almmo0 import ALMMo0
import pandas as pd
from mafese import Data
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
from pyfume.EstimateConsequentParameters import ConsequentEstimator
from pyfume.SimpfulModelBuilder import SugenoFISBuilder
from pyfume.Tester import SugenoFISTester
from mafese import UnsupervisedSelector, FilterSelector, LassoSelector, TreeSelector
from mafese import SequentialSelector, RecursiveSelector, MhaSelector, MultiMhaSelector
import numpy as np
from numpy import clip, column_stack, argmax
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score

# %% Open the data as a Dataframe

flight_Data = pd.read_csv('train.csv')

# %% Exclude some "a priori" features and get dummies with others

flight_Data = flight_Data.drop(['DEP_TIME_BLK', 'PREVIOUS_AIRPORT'], axis = 1)
train_data = flight_Data

flight_Data = pd.get_dummies(flight_Data, columns=['CARRIER_NAME'])
flight_Data =  pd.get_dummies(flight_Data, columns=['DEPARTING_AIRPORT'])

# %% Random sampling 

delayed_data = flight_Data[flight_Data['DEP_DEL15'] == 1]
delayed_data = delayed_data.sample(n=5000, random_state=42)
ontime_data =flight_Data[flight_Data['DEP_DEL15'] == 0]
ontime_data = ontime_data.sample(n=len(delayed_data), random_state=42)

flight_Data = pd.concat([ontime_data, delayed_data], axis=0, ignore_index=True)

# %%  Pre processing of data (Removing duplicate data, NDA, etc...)

flight_Data = flight_Data.dropna()
flight_Data = flight_Data.drop_duplicates()

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

# %% Modelling

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ALMMo0()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('<===IRIS DATASET===>:\nAccuracy: {}'.format(\
     round(accuracy_score(Y_test, Y_pred),3)))