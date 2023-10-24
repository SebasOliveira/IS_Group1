# %% Import packages
from almmo0 import ALMMo0
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %% Iris dataset demo
data_iris = load_iris()
X = data_iris.data
Y = data_iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = ALMMo0()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print('<===IRIS DATASET===>:\nAccuracy: {}'.format(\
     round(accuracy_score(Y_test, Y_pred),3)))
    
# %% Breast cancer dataset demo
data_cancer = load_breast_cancer()
X = data_cancer.data
Y = data_cancer.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = ALMMo0()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print('<===BREAST CANCER DATASET===>:\nAccuracy: {}'.format(\
     round(accuracy_score(Y_test, Y_pred),3)))
