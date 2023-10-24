import numpy as np
import pandas as pd
import math

# Initial cloud radius
r0 = math.sqrt(2*(1-math.cos(math.radians(15))))

class ALMMo0:
    def __init__(self):
        self.feature_names = []
        self.target_table = []
        self.class_models = []
        
    def fit(self, X, Y, feature_names=None, target_names=None):
        if feature_names != None:
            self.feature_names = feature_names
        else:
            self.feature_names = ['var' + str(i+1) for i in range(0, X.shape[1])]
            
        if target_names != None:
            self.target_table = pd.DataFrame(data = target_names,\
                                             columns = ['label'])
            self.target_table['index'] = np.arange(len(target_names))
            self.target_table['value'] = np.unique(Y)
        else:
            self.target_table = pd.DataFrame(data = np.unique(Y),\
                                             columns = ['label'])
            self.target_table['index'] = np.arange(self.target_table.shape[0])
            
        # Split data by class
        X_class = [X[Y == self.target_table.loc[i,'label'],:] for i in range(0,self.target_table.shape[0])]
        
        # Initialize class models
        self.class_models = [ALMMo0ClassModel() for _ in range(0, len(X_class))]
        
        # Train each class model
        for c in range(0, len(X_class)):
            for k in range(0, X_class[c].shape[0]):
                # Normalize sample
                x_k_c = (X_class[c][k,:]/np.linalg.norm(X_class[c][k,:])).reshape(-1,1)
                self.class_models[c].update(x_k_c)
                
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for k in range(0, y_pred.size):
            x = (X[k,:]/np.linalg.norm(X[k,:])).reshape(-1,1)
            k_lambdas = np.zeros(len(self.class_models))
            for c in range(0, k_lambdas.size):
                k_lambdas[c] = np.max(self.class_models[c].compute_lambdas(x))
            y_pred[k] = self.target_table.loc[np.argmax(k_lambdas),'label']
                
        return y_pred
    
    
class ALMMo0ClassModel:
    def __init__(self):
        # Global parameters
        self.K = 0 # number of samples
        self.Mu = [] # global mean
        self.X = 0 # global average scalar product
        # Cloud parameters
        self.F = [] # cloud focal points
        self.N = 0 # number of clouds
        self.Mc = [] # number of cloud members
        self.Rc = [] # clouds radius
        
    def update(self, x):
        self.K += 1 # increment number of samples
        if self.K == 1:
            # Initialize global parameters
            self.Mu = x
            self.X = np.power(np.linalg.norm(x),2)
            # Initialize cloud parameters
            self.F = x
            self.N = 1
            self.Mc = np.array([1])
            self.Rc = np.array([r0])
        else:
            # Update global parameters
            self.Mu = ((self.K-1)/self.K)*self.Mu + (1/self.K)*x
            self.X = ((self.K-1)/self.K)*self.X + (1/self.K)*np.power(np.linalg.norm(x),2)
            # Compute unimodal density of x relative to global mean
            umd_Mu = unimodal_density(x, self.Mu, self.X)
            umd_F = unimodal_density(self.F, self.Mu, self.X)
            # Check density condition
            if umd_Mu > np.max(umd_F) or umd_Mu < np.min(umd_F):
                # Initialize new cloud
                self.F = np.column_stack((self.F, x))
                self.N += 1
                self.Mc = np.append(self.Mc, 1)
                self.Rc = np.append(self.Rc, r0)
            else:
                # Find nearest cloud to x
                c_near = np.argmin(np.linalg.norm(x-self.F, axis=0))
                F_near = self.F[:,c_near].reshape(-1,1)
                # Check distance condition
                if np.linalg.norm(x-F_near) <= self.Rc[c_near]:
                    # Update nearest cloud
                    self.Mc[c_near] += 1
                    self.F[:,c_near] = (((self.Mc[c_near]-1)/self.Mc[c_near])*F_near + (1/self.Mc[c_near])*x).reshape(-1,)
                    self.Rc[c_near] = math.sqrt(0.5*(np.power(self.Rc[c_near],2)+(1-np.power(np.linalg.norm(F_near),2))))
                else:
                    # Initialize new cloud
                    self.F = np.column_stack((self.F, x))
                    self.N += 1
                    self.Mc = np.append(self.Mc, 1)
                    self.Rc = np.append(self.Rc, r0)
                    
    def compute_lambdas(self, x):
        return np.exp(-0.5*np.linalg.norm(x-self.F,axis=0))
            
            
def unimodal_density(x, mu, X):
    a = np.power(np.linalg.norm(x-mu,axis=0),2)
    b = X - np.power(np.linalg.norm(mu),2)
    D = 1/(1+a/b)

    return D
            
            
        