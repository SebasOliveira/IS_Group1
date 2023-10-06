# %% Imports
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
test_values = []

test = 0

for i in range(0,9):
    
    test = test + 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=12)


    regr = MLPClassifier(hidden_layer_sizes=(80,80),random_state=42, max_iter=50000)
    regr.fit(X_train, y_train)


    y_pred = regr.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    accuracy_values.append(acc_score)
    test_values.append(test)
  
    print("Accuracy: {:.3f}".format(acc_score))
    kappa = cohen_kappa_score(y_test, y_pred)
    print("Kappa Score: {:.3f}".format(kappa))

#%% Plot
plt.plot(test_values, accuracy_values, marker='o')
plt.title('Accuracy Vs Train/Test Size Partition')
plt.xlabel('Test Size Percentage')
plt.ylabel('Accuracy')
plt.show()