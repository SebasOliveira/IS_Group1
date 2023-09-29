# %% Imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pyfume.Clustering import Clusterer
from pyfume.EstimateAntecendentSet import AntecedentEstimator
from pyfume.EstimateConsequentParameters import ConsequentEstimator
from pyfume.SimpfulModelBuilder import SugenoFISBuilder
from pyfume.Tester import SugenoFISTester
from sklearn.metrics import accuracy_score, cohen_kappa_score
from numpy import clip, column_stack, argmax, vectorize
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
# %% Load dataset and create train-test sets
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# %% Train 0 vs all model

# Create 0-vs-all target vector
y_train_0_vs_all = vectorize({0:1,1:0,2:0}.get)(y_train)
# Cluster the input-output space
cl = Clusterer(x_train=X_train, y_train=y_train_0_vs_all, nr_clus=8)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimate membership functions parameters
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimate consequent parameters
ce = ConsequentEstimator(X_train, y_train_0_vs_all, part_matrix)
conseq_params = ce.suglms()
# Build first-order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_0_vs_all = modbuilder.get_model()

# %% Train 1 vs all model

# Create 1-vs-all target vector
y_train_1_vs_all = vectorize({0:0,1:1,2:0}.get)(y_train)
# Cluster the input-output space
cl = Clusterer(x_train=X_train, y_train=y_train_1_vs_all, nr_clus=8)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimate membership functions parameters
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimate consequent parameters
ce = ConsequentEstimator(X_train, y_train_1_vs_all, part_matrix)
conseq_params = ce.suglms()
# Build first-order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_1_vs_all = modbuilder.get_model()


# %% Train 2 vs all model

# Create 2-vs-all target vector
y_train_2_vs_all = vectorize({0:0,1:0,2:1}.get)(y_train)
# Cluster the input-output space
cl = Clusterer(x_train=X_train, y_train=y_train_2_vs_all, nr_clus=8)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimate membership functions parameters
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimate consequent parameters
ce = ConsequentEstimator(X_train, y_train_2_vs_all, part_matrix)
conseq_params = ce.suglms()
# Build first-order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_2_vs_all = modbuilder.get_model()

# %% Get class probabilities predictions for each ova model

# Class probabilities predictions for 0 vs all
modtester = SugenoFISTester(model_0_vs_all, X_test, var_names)
y_pred_probs_0_vs_all = clip(modtester.predict()[0], 0, 1)
y_pred_probs_0_vs_all = column_stack((1 - y_pred_probs_0_vs_all, y_pred_probs_0_vs_all))

# Class probabilities predictions for 1 vs all
modtester = SugenoFISTester(model_1_vs_all, X_test, var_names)
y_pred_probs_1_vs_all = clip(modtester.predict()[0], 0, 1)
y_pred_probs_1_vs_all = column_stack((1 - y_pred_probs_1_vs_all, y_pred_probs_1_vs_all))

# Class probabilities predictions for 2 vs all
modtester = SugenoFISTester(model_2_vs_all, X_test, var_names)
y_pred_probs_2_vs_all = clip(modtester.predict()[0], 0, 1)
y_pred_probs_2_vs_all = column_stack((1 - y_pred_probs_2_vs_all, y_pred_probs_2_vs_all))

# %% Aggregate class probabilities and get class predictions

y_pred_probs = column_stack((y_pred_probs_0_vs_all[:,1],y_pred_probs_0_vs_all[:,0],y_pred_probs_0_vs_all[:,0])) +\
               column_stack((y_pred_probs_1_vs_all[:,0],y_pred_probs_1_vs_all[:,1],y_pred_probs_1_vs_all[:,0])) +\
               column_stack((y_pred_probs_2_vs_all[:,0],y_pred_probs_2_vs_all[:,0],y_pred_probs_2_vs_all[:,1]))
y_pred_probs = y_pred_probs/y_pred_probs.sum(axis=1,keepdims=1)

y_pred = argmax(y_pred_probs,axis=1)

cm = confusion_matrix(y_test, y_pred)

# %% Compute classification metrics
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc:.4f}")


# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()