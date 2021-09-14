from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn import utils
from random import seed
from hyperopt import hp, tpe, STATUS_OK, Trials, fmin
from xgboost import XGBRegressor
import numpy as np 
import pickle 
import warnings
warnings.filterwarnings("ignore") 

class CogniHyperOptimize:
    ''' Hyper Optimization technique which uses random search internally having an objective
    fucntion and its related optimization method'''

    def __init__(self,param_grid,model_optimize = None,h_best = None):
        self.param_grid = param_grid
        self.h_best = h_best
        self.model_optimize = model_optimize
    

    # Hyper Opt objective function defined 
    def objective_hopt(self,param):
        f_file ="/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt"
        # f_file = "feature_train.txt"
        l_file ="/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt"
        # l_file = "label_train.txt"
        with open(f_file,'rb') as feature_file:
            features = pickle.load(feature_file)
        with open(l_file,'rb') as label_file:
            labels = pickle.load(label_file)
        with open(self.model_optimize,'rb') as model_file: 
            m_name = pickle.load(model_file)
        param = self.param_grid
        score = cross_val_score(m_name,features,labels).mean()
        # print("Display for HyperOpt : {}".format(score))
        return(score)
    
    # Hyperopt parameter space 
    def hyperopt(self,max_evals):
        space = {}
        for key,value in self.param_grid.items():
            first_param = sorted(value)[0]
            last_param = sorted(value)[-1]  
            j = {key : hp.quniform(key ,first_param,last_param,0.1)}
            space.update(j)
        self.h_best = fmin(fn=self.objective_hopt,space=space,algo=tpe.suggest,max_evals=max_evals,verbose=0)
        # return(self.h_best)
        print("The best parameters-> by Hyper Opt {}".format(self.h_best))