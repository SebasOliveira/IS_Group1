# %% Imports

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from pyfume.Clustering import Clusterer
from pyfume.EstimateAntecendentSet import AntecedentEstimator
from pyfume.EstimateConsequentParameters import ConsequentEstimator
from pyfume.SimpfulModelBuilder import SugenoFISBuilder
from pyfume.Tester import SugenoFISTester
from numpy import copy
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from numpy import clip, column_stack, argmax,vectorize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
 #%% Load dataset and Normalize Data
 
column_names = ['Tipo', 'Alcohol','Malic acid','Ash','Alcalinity of ash',\
                'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',\
                'Proanthocyanins','Color intensity','Hue','OD280OD315','Proline'] 
    
column_names = [column_names[i].title().replace(' ','') for i in range(0, len(column_names))]    

var_names = [item for item in column_names if item != 'Tipo']   
 
wine_data = pd.read_csv('wine.data',names=column_names) 

y = wine_data['Tipo'] 
X = wine_data.drop(columns=['Tipo']) 

y = y.values-1 
X = X.values 

#scaler = MinMaxScaler()
#scaler = StandardScaler()
#X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=70)

#%%
accuracy_values = []

# Iterate over the number of clusters from 3 to 20
for num_clusters in range(3, 21):
    
    num = num_clusters
    # Train 0 vs all model
    y_train_0_vs_all = vectorize({0:1,1:0,2:0}.get)(y_train)
    cl = Clusterer(x_train=X_train, y_train=y_train_0_vs_all, nr_clus=num)
    clust_centers, part_matrix, _ = cl.cluster(method='fcm')
    ae = AntecedentEstimator(X_train, part_matrix)
    antecedent_params = ae.determineMF()
    ce = ConsequentEstimator(X_train, y_train_0_vs_all, part_matrix)
    conseq_params = ce.suglms()
    modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
    model_0_vs_all = modbuilder.get_model()

    # Train 1 vs all model
    y_train_1_vs_all = vectorize({0:0,1:1,2:0}.get)(y_train)
    cl = Clusterer(x_train=X_train, y_train=y_train_1_vs_all, nr_clus=num)
    clust_centers, part_matrix, _ = cl.cluster(method='fcm')
    ae = AntecedentEstimator(X_train, part_matrix)
    antecedent_params = ae.determineMF()
    ce = ConsequentEstimator(X_train, y_train_1_vs_all, part_matrix)
    conseq_params = ce.suglms()
    modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
    model_1_vs_all = modbuilder.get_model()

    # Train 2 vs all model
    y_train_2_vs_all = vectorize({0:0,1:0,2:1}.get)(y_train)
    cl = Clusterer(x_train=X_train, y_train=y_train_2_vs_all, nr_clus=num)
    clust_centers, part_matrix, _ = cl.cluster(method='fcm')
    ae = AntecedentEstimator(X_train, part_matrix)
    antecedent_params = ae.determineMF()
    ce = ConsequentEstimator(X_train, y_train_2_vs_all, part_matrix)
    conseq_params = ce.suglms()
    modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
    model_2_vs_all = modbuilder.get_model()
 
    
    modtester = SugenoFISTester(model_0_vs_all, X_test, var_names)
    y_pred_probs_0_vs_all = clip(modtester.predict()[0], 0, 1)
    y_pred_probs_0_vs_all = column_stack((1 - y_pred_probs_0_vs_all, y_pred_probs_0_vs_all))

    modtester = SugenoFISTester(model_1_vs_all, X_test, var_names)
    y_pred_probs_1_vs_all = clip(modtester.predict()[0], 0, 1)
    y_pred_probs_1_vs_all = column_stack((1 - y_pred_probs_1_vs_all, y_pred_probs_1_vs_all))

    modtester = SugenoFISTester(model_2_vs_all, X_test, var_names)
    y_pred_probs_2_vs_all = clip(modtester.predict()[0], 0, 1)
    y_pred_probs_2_vs_all = column_stack((1 - y_pred_probs_2_vs_all, y_pred_probs_2_vs_all))

    # Aggregate class probabilities and get class predictions
    y_pred_probs = (
        column_stack((y_pred_probs_0_vs_all[:,1], y_pred_probs_0_vs_all[:,0], y_pred_probs_0_vs_all[:,0])) +
        column_stack((y_pred_probs_1_vs_all[:,0], y_pred_probs_1_vs_all[:,1], y_pred_probs_1_vs_all[:,0])) +
        column_stack((y_pred_probs_2_vs_all[:,0], y_pred_probs_2_vs_all[:,0], y_pred_probs_2_vs_all[:,1]))
    )
    y_pred_probs = y_pred_probs / y_pred_probs.sum(axis=1, keepdims=1)

    y_pred = argmax(y_pred_probs, axis=1)

    # Compute and print classification metrics
    acc_score = accuracy_score(y_test, y_pred)
    print(f"Number of Clusters: {num_clusters}, Accuracy: {acc_score}")
    
    # Save the accuracy value for plotting
    accuracy_values.append(acc_score)



#%%
plt.plot(range(3, 21), accuracy_values, marker='o')
plt.title('Accuracy vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Accuracy')
plt.show()


