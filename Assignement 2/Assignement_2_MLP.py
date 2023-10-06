# %% Imports
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

# %% Load dataset and create train-test sets
data = load_iris()
X = data.data
y = data.target
var_names = data.feature_names
var_names = [var_names[i][0:-5] for i in range(0, len(var_names))]
var_names = [var_names[i].title().replace(' ','') for i in range(0, len(var_names))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# %% Train model
regr = MLPClassifier(hidden_layer_sizes=(10,10),random_state=42, max_iter=500)
regr.fit(X_train, y_train)

# %% Get model predictions
y_pred = regr.predict(X_test)

# %% Compute classification metrics
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))

