# %% Imports

import loader as ld
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from pyfume.Clustering import Clusterer
from pyfume.EstimateAntecendentSet import AntecedentEstimator
from pyfume.EstimateConsequentParameters import ConsequentEstimator
from pyfume.SimpfulModelBuilder import SugenoFISBuilder
from pyfume.Tester import SugenoFISTester
from numpy import clip
import ACO as aco
import Flight_Model as cm
import gc
import math

# %% Open and read the file

loader = ld.loader()
 
flight_Data = loader.pdLoadFlight("train.csv")

# %%  Factorize the Carriers Names and Departure Airports

#flight_Data = pd.get_dummies(flight_Data, columns=['CARRIER_NAME'])
#flight_Data =  pd.get_dummies(flight_Data, columns=['DEPARTING_AIRPORT'])


# %%
flight_Data = flight_Data.drop(['DEP_TIME_BLK', 'PREVIOUS_AIRPORT', 'CARRIER_NAME', 'DEPARTING_AIRPORT'], axis = 1)
train_data = flight_Data

# %% 
delayed_data = train_data[train_data['DEP_DEL15'] == 1]
delayed_data = delayed_data.sample(n=5000, random_state=42)
ontime_data = train_data[train_data['DEP_DEL15'] == 0]
ontime_data = ontime_data.sample(n=len(delayed_data), random_state=42)

train_data = pd.concat([ontime_data, delayed_data], axis=0, ignore_index=True)

# %%
train_data.columns = [f'V{i}' for i in range(1, len(train_data.columns) + 1)]
column_names = train_data.columns.tolist()


# %%
acoModel = aco.ACO(train_data,maxIteration=1000,antNumber=100,cc=1,Q=0.1,e=0.95)

# %%
result = acoModel.simulate()
# %% Feature Selection (ACO)

acoModel = aco.ACO(train_data,maxIteration=100,antNumber=100,cc=1,Q=0.1,e=0.95)
result = acoModel.simulate()

# %% Unwanted Collumns and feature selection

unwanted_columns = ['DEP_TIME_BLK', 'SEGMENT_NUMBER', 'PREVIOUS_AIRPORT']

train_data = train_data.drop(columns=unwanted_columns)

X_train = train_data.drop(columns=['DEP_DEL15']) 
y_train = train_data['DEP_DEL15'] 

y_train = y_train.values
X_train = X_train.values 

train_data.columns = [f'V{i}' for i in range(1, len(train_data.columns) + 1)]
column_names = train_data.columns.tolist()
column_names = [item for item in column_names if item != 'V3']  

# %% Normalize the Data

scaler = MinMaxScaler()
#scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42097)

# %% Cluster the input-output space
cl = Clusterer(x_train=X_train, y_train=y_train, nr_clus=10)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')

# %% Estimate membership functions parameters
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()

# %% Estimate consequent parameters
ce = ConsequentEstimator(X_train, y_train, part_matrix)
conseq_params = ce.suglms()

# %% Build first-order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, column_names, save_simpful_code=False)
model = modbuilder.get_model()

# %% Get model predictions
modtester = SugenoFISTester(model, X_val, column_names)
y_pred_probs = clip(modtester.predict()[0], 0, 1)
y_pred_probs = column_stack((1 - y_pred_probs, y_pred_probs))
y_pred = argmax(y_pred_probs,axis=1)