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

flight_Data = flight_Data.drop(['DEP_TIME_BLK', 'PREVIOUS_AIRPORT'], axis = 1)

# %% One hot Encoding
flight_Data = pd.get_dummies(flight_Data, columns=['CARRIER_NAME'])
num_variables_created_2 = flight_Data.shape[1]

flight_Data =  pd.get_dummies(flight_Data, columns=['DEPARTING_AIRPORT'])
num_variables_created_3 = flight_Data.shape[1]

# %% Embedding

from tensorflow.keras.layers import Input, Embedding, Flatten
from tensorflow.keras.models import Model

# Encoding the categorical features as integers
flight_Data['EncodedCategory1'] = flight_Data['CARRIER_NAME'].astype('category').cat.codes
flight_Data['EncodedCategory2'] = flight_Data['DEPARTING_AIRPORT'].astype('category').cat.codes

# Determine the number of unique categories and embedding size
num_categories1 = len(flight_Data['EncodedCategory1'].unique())
num_categories2 = len(flight_Data['EncodedCategory2'].unique())
embedding_size = 17  # Adjust the embedding size as needed
embedding_size2 = 10

# Define input layers for each categorical feature
input_category1 = Input(shape=(1,))
input_category2 = Input(shape=(1,))

# Define embedding layers for each feature
embedding_category1 = Embedding(input_dim=num_categories1, output_dim=embedding_size)(input_category1)
embedding_category2 = Embedding(input_dim=num_categories2, output_dim=embedding_size2)(input_category2)

flattened_category1 = Flatten()(embedding_category1)
flattened_category2 = Flatten()(embedding_category2)

# Create models for each feature
model_category1 = Model(inputs=input_category1, outputs=flattened_category1)
model_category2 = Model(inputs=input_category2, outputs=flattened_category2)

# Compile the models (not needed for data preprocessing)

# Extract embeddings for the entire dataset
embeddings_category1 = model_category1.predict(flight_Data['EncodedCategory1'])
embeddings_category2 = model_category2.predict(flight_Data['EncodedCategory2'])


num_columns = len(embeddings_category1[0])
column_names = [f'E{i+1}' for i in range(num_columns)]
df = pd.DataFrame(embeddings_category1, columns=column_names)

num_columns2 = len(embeddings_category2[0])
column_names2 = [f'F{i+1}' for i in range(num_columns2)]
df2 = pd.DataFrame(embeddings_category2, columns=column_names2)


flight_Data = flight_Data.drop(['CARRIER_NAME', 'DEPARTING_AIRPORT'], axis = 1)

flight_Data = pd.concat([flight_Data,df,df2], axis=1)
# %% Random sampling 

delayed_data = flight_Data[flight_Data['DEP_DEL15'] == 1]
delayed_data = delayed_data.sample(n=5000, random_state=120)
ontime_data =flight_Data[flight_Data['DEP_DEL15'] == 0]
ontime_data = ontime_data.sample(n=len(delayed_data), random_state=120)

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

# %%  PCA 
from sklearn.decomposition import PCA

n_components = 50 
pca = PCA(n_components=n_components)
X= pca.fit_transform(X_scaled)
V = 'V'
column_names =  [f'{V}{i}' for i in range(1, X.shape[1] + 1)]
# %% Autoencoders

# Define the dimensions of your input data
input_dim = X_scaled.shape[1]  # Use the actual dimension of your data
encoding_dim = 20 # Adjust as needed

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation="sigmoid")(encoded)
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoded)
encoder =  tf.keras.models.Model(inputs=input_layer, outputs=encoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])  # For binary classification

num_epochs = 200
batch_size = 32

autoencoder.fit(X_scaled, X_scaled, epochs=num_epochs, batch_size=batch_size)
encoded_features = encoder.predict(X_scaled)
X = encoded_features
V = 'V'
column_names =  [f'{V}{i}' for i in range(1, X.shape[1] + 1)]

# %%  Feature Selection

X_scaled = X_scaled.values
data = Data(X_scaled, y)
data.split_train_test(test_size=0.2, inplace=True)

# Feature selector with a Tree Structure
#feat_selector = TreeSelector(problem="classification", estimator="tree")

# Feature selector witha  Filter Method
#feat_selector = FilterSelector(problem='classification', method='SPEARMAN', n_features = 10)

#feat_selector = SequentialSelector(problem="classification", estimator="rf", n_features=20, direction="forward")
#feat_selector = RecursiveSelector(problem="classification", estimator="rf", n_features=30)

# Feature selector with a Metaheuristic Algorithm
feat_selector = MhaSelector(problem="classification", estimator="knn", optimizer="OriginalPSO", optimizer_paras=None, transfer_func="vstf_01", obj_name="AS")
feat_selector.fit(data.X_train, data.y_train)

feat_selector.fit(data.X_train, data.y_train)
print(feat_selector.selected_feature_masks)
print(feat_selector.selected_feature_solution)
selected_indices = np.where(feat_selector.selected_feature_solution == 1)[0]

X =X_scaled[:, selected_indices]


 # %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42097)

cl = Clusterer(x_train=X_train, y_train=y_train, nr_clus=25)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')


ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()

ce = ConsequentEstimator(X_train, y_train, part_matrix)
conseq_params = ce.suglms()

# %% Build the Takagi Sugeno Model and Calculate the probabilities

selected_column_names = [column_names[i] for i in selected_indices]
#selected_column_names = column_names

modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, selected_column_names, save_simpful_code=False)
model = modbuilder.get_model()

modtester = SugenoFISTester(model, X_val, selected_column_names)
y_pred_probs = clip(modtester.predict()[0], 0, 1)
y_pred_probs = column_stack((1 - y_pred_probs, y_pred_probs))
y_pred = argmax(y_pred_probs,axis=1)

# %% Calculate the Performance Indices

acc_score = accuracy_score(y_val, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
rec_score = recall_score(y_val, y_pred)
print("Recall: {:.3f}".format(rec_score))
prec_score = precision_score(y_val, y_pred)
print("Precision Score: {:.3f}".format(prec_score))
F1_score = f1_score(y_val, y_pred)
print("F1-Score: {:.3f}".format(F1_score))
kappa = cohen_kappa_score(y_val, y_pred)
print("Kappa Score: {:.3f}".format(kappa))


