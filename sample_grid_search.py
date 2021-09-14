import pandas as pd 
import numpy as np 
from data_preprocess import DataPreprocessing
from metrics import MetricsCal
from metrics_class import MetricsClass
from sklearn import preprocessing
from sklearn import utils
from random import seed
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
warnings.filterwarnings("ignore")

class CogniGridSearch: 

    ''' Grid Search optimization where the feature train and label train files are loaded'''

    def __init__(self,param_grid,feature_test,label_test,model_optimize = None,g_best = None):


        self.param_grid = param_grid
        self.g_best = g_best 
        self.model_optimize = model_optimize
        self.label_test = label_test
        self.feature_test = feature_test

    
    # Grid Search Optimization with three files picked Feature Train, Label Train, Model 
    def grid_search_cv(self,cv):
        f_file = "/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt"
        # f_file = "feature_train.txt"
        l_file = "/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt"
        # l_file = "label_train.txt" 
        with open(f_file,'rb') as feature_file:
            features = pickle.load(feature_file)
        with open(l_file,'rb') as label_file:
            labels = pickle.load(label_file) 
        with open(self.model_optimize,'rb') as model_file: 
            m_name = pickle.load(model_file) 
         
        grid_search = GridSearchCV(estimator = m_name, param_grid = self.param_grid, n_jobs= None, cv= cv, verbose=0, iid = 'False')
        gxb = grid_search.fit(features,labels)
        self.g_best = gxb.best_params_
        # return(self.g_best)
        print("The best parameters-> by grid search:{}".format(self.g_best))         
        
    # New parameters are used for new accuracy metrics calculation 
    def optimize_metrics(self,**param):
        
        ''' Best parameters are stored in the form of dictionary and thus are unpacked 
        which are eventually loaded into a list. Then the feature and label files are unpickled.
        The new list of parameters are entered in the model to get trained and further the
        features and labels are fitted to get new predicted labels and thus a metrics is further 
        calculated.
        '''

        param = self.g_best

        lst = []
        for key,value in param.items():
            s = '{} = {}'
            l = s.format(key,value)
            lst.append(l)
        
        f_file = "/home/congnitensor/Python_projects/model_class/file_logs/feature_train.txt"
        # f_file = "feature_train.txt"
        l_file = "/home/congnitensor/Python_projects/model_class/file_logs/label_train.txt"
        # l_file = "label_train.txt" 
        with open(f_file,'rb') as feature_file:
            features = pickle.load(feature_file)
        with open(l_file,'rb') as label_file:
            labels = pickle.load(label_file) 
        with open(self.model_optimize,'rb') as model_file: 
            m_name = pickle.load(model_file) 
     
        print("Old list :", lst)
        print("New unpacked list : ", *lst)

        model = m_name
    
        print(model)
        model.fit(features,labels) 
        label_predicted = model.predict(self.feature_test)
        print(label_predicted) 
        





        
    