from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
import pandas as pd
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


accuracy_values = []
layer_sizes_values = []

hl = 2
hl_2 = 0

for i in range(1,15):
    
    hl_2 = hl_2 + 10
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=70)


    regr = MLPClassifier(hidden_layer_sizes=(hl_2,hl_2,hl_2),random_state=42, max_iter=5000)
    regr.fit(X_train, y_train)


    y_pred = regr.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    accuracy_values.append(acc_score)
    layer_sizes_values.append(hl_2)
  
    print("Accuracy: {:.3f}".format(acc_score))
    kappa = cohen_kappa_score(y_test, y_pred)
    print("Kappa Score: {:.3f}".format(kappa))
    
#%% Plot
plt.plot(layer_sizes_values, accuracy_values, marker='o')
plt.title('Accuracy Vs hidden Layer Sizes')
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy')
plt.show()    
