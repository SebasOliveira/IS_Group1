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

# %%
flight_Data = pd.read_csv('train.csv')

# %% Get dummy Variables

flight_Data = pd.get_dummies(flight_Data, columns=['CARRIER_NAME'])
flight_Data =  pd.get_dummies(flight_Data, columns=['DEPARTING_AIRPORT'])

# %%
flight_Data = flight_Data.drop(['DEP_TIME_BLK', 'PREVIOUS_AIRPORT'], axis = 1)
train_data = flight_Data

# %% 
delayed_data = train_data[train_data['DEP_DEL15'] == 1]
delayed_data = delayed_data.sample(n=5000, random_state=231)
ontime_data = train_data[train_data['DEP_DEL15'] == 0]
ontime_data = ontime_data.sample(n=len(delayed_data), random_state=8199)

train_data = pd.concat([ontime_data, delayed_data], axis=0, ignore_index=True)

# %% Data Understanding

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [50, 60]
# Scatter plots
train_data.plot(kind="scatter", x="NUMBER_OF_SEATS", y="DEP_DEL15")

# Histogram plots
train_data.hist()
plt.show()

from pandas.plotting import scatter_matrix
# Scatter plot matrix
scatter_matrix(train_data)
plt.show()

# %% Data Preparation

train_data = train_data.dropna()
train_data = train_data.drop_duplicates()

# %%
train_data.columns = [f'V{i}' for i in range(1, len(train_data.columns) + 1)]
column_names = train_data.columns.tolist()
column_names = [item for item in column_names if item != 'V3']  


# %% Normalizing the Dta

accuracy_values = []

for num_features in range(3, 20):
    
    X = train_data.drop(columns=['V3']) 
    y = train_data['V3'] 

    y = y.values
    X = X.values 
    1
    data = Data(X, y)
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    from mafese.embedded.tree import TreeSelector


    data.split_train_test(test_size=0.2, inplace=True)
    #feat_selector = TreeSelector(problem="classification", estimator="tree")
    feat_selector = FilterSelector(problem='classification', method='SPEARMAN', n_features = 20)
    #feat_selector = MhaSelector(problem="classification", estimator="knn",
                            #optimizer="OriginalACOR", optimizer_paras=None,
                            #transfer_func="vstf_01", obj_name="AS")
    feat_selector.fit(data.X_train, data.y_train)
    print(feat_selector.selected_feature_masks)
    print(feat_selector.selected_feature_solution)
    selected_indices = np.where(feat_selector.selected_feature_solution == 1)[0]
    X =X[:, selected_indices]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=212121)

    cl = Clusterer(x_train=X_train, y_train=y_train, nr_clus=num_features)
    clust_centers, part_matrix, _ = cl.cluster(method='fcm')


    ae = AntecedentEstimator(X_train, part_matrix)
    antecedent_params = ae.determineMF()

    ce = ConsequentEstimator(X_train, y_train, part_matrix)
    conseq_params = ce.suglms()


    selected_column_names = [column_names[i] for i in selected_indices]


    modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, selected_column_names, save_simpful_code=False)
    model = modbuilder.get_model()

    modtester = SugenoFISTester(model, X_val, selected_column_names)
    y_pred_probs = clip(modtester.predict()[0], 0, 1)
    y_pred_probs = column_stack((1 - y_pred_probs, y_pred_probs))
    y_pred = argmax(y_pred_probs,axis=1)

    acc_score = accuracy_score(y_val, y_pred)
    
    accuracy_values.append(acc_score)

# %%    
import matplotlib.pyplot as plt

plt.plot(range(3, 20), accuracy_values, marker='o')   
plt.xlabel('Number of Clusters')
plt.ylabel('Accuracy')
print("Accuracy: {:.3f}".format(acc_score))
rec_score = recall_score(y_val, y_pred)
print("Recall: {:.3f}".format(rec_score))
prec_score = precision_score(y_val, y_pred)
print("Precision Score: {:.3f}".format(prec_score))
F1_score = f1_score(y_val, y_pred)
print("F1-Score: {:.3f}".format(F1_score))
kappa = cohen_kappa_score(y_val, y_pred)
print("Kappa Score: {:.3f}".format(kappa))

