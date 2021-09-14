from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn import utils
from random import seed
from data_preprocess import DataPreprocessing
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
import numpy as np 
import pickle
import warnings
warnings.filterwarnings("ignore")

class CogniBayesOptimize:
    
    '''Bayesian Optimization where we have defined an objective function and its related 
    optimization method. '''

    def __init__(self,param_grid,model_optimize = None,b_best = None):
        
        self.param_grid = param_grid
        self.b_best = b_best
        self.model_optimize = model_optimize

        
    # Bayes Opt objective Function with cross validation score mean 
    def objective_bayes(self,**param):
        f_file = "/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt"
        # f_file = "feature_train.txt"
        l_file = "/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt"
        # l_file = "label_train.txt"
        with open(f_file,'rb') as feature_file:
            features = pickle.load(feature_file)
        with open(l_file,'rb') as label_file:
            labels = pickle.load(label_file)
        with open(self.model_optimize, 'rb') as model_file:
            m_name = pickle.load(model_file) 
        param = self.param_grid
        
        score = cross_val_score(m_name,features, labels).mean()
        # print("Display for Bayes Opt : {}".format(score))
        return(score)
    
    # Bayes Opt parameter boundary space defined  
    def bayesoptimize(self,init_points,n_iter):
        pbounds = {}
        for key,value in self.param_grid.items():
            first_param = sorted(value)[0]
            last_param = sorted(value)[-1]  
            j = {key : (first_param,last_param)}
            pbounds.update(j)
        b_optimizer = BayesianOptimization(f=self.objective_bayes,pbounds=pbounds,random_state=1, verbose = 0)
        b_optimizer.maximize(init_points = init_points , n_iter = n_iter)
        self.b_best = b_optimizer.max
        # return(self.b_best)
        print("The best parameters-> by Bayes Opt {}".format(self.b_best)) 

        for p in self.b_best['params']:
            print(p) 
            
        