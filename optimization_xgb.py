from data_preprocess import DataPreprocessing
from regress_train import RegressionBase
from class_train import ClassificationBase 
from grid_search import CogniGridSearch
from bayesopt import CogniBayesOptimize
from hoptimize import CogniHyperOptimize 
import pickle
import argparse 


class XgbOptimizeMethod:
    
    '''Optimization base class calling other optimization technique namely Grid Search , 
    Bayes Opt and Hyper Opt. Here this class specifically defined for xgboost with parameters
    as n_estimiators , max_depth, colsample_bytree, gamma and learning rate. Cross Validation
    score is used for grid search, maximum evaluations is used for hyperopt and number of iteration
     as well as initial points used for Bayesian Optimization.'''
    
    def __init__(self,**kwargs):
        
        if 'params' in kwargs:
            self.params = kwargs['params']
        if 'init_points' in kwargs:
            self.init_points = kwargs['init_points']
        if 'n_iter' in kwargs:
            self.n_iter = kwargs['n_iter']
        if 'max_evals' in kwargs:
            self.max_evals = kwargs['max_evals']
        if 'cross_val' in kwargs:
            self.cross_val = kwargs['cross_val']
        if  'model_path' in kwargs:
            self.model_path = kwargs['model_path']
    
    # Calling Grid search for XGBoost
    def grid_search(self):

        obj_1 = CogniGridSearch(self.params,model_optimize = self.model_path)
        obj_1.grid_search_cv(self.cross_val)
    
    # Calling Bayesian Optimization for XGBoost
    def bayes_opt(self):
    
        obj_2 = CogniBayesOptimize(self.params,model_optimize = self.model_path)
        obj_2.objective_bayes()
        obj_2.bayesoptimize(self.init_points,self.n_iter)
    
    # Calling Hyper Optimization for XGBoost
    def hyper_opt(self):
        obj_3 = CogniHyperOptimize(self.params,model_optimize= self.model_path)
        obj_3.objective_hopt(self.params)
        obj_3.hyperopt(self.max_evals)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-n_estimators", nargs = '+', type = int)
    parser.add_argument("-max_depth", nargs = '+', type = int)
    parser.add_argument("-colsample_bytree", nargs = '+', type = float)
    parser.add_argument("-gamma", nargs = '+', type = float)
    parser.add_argument("-learning_rate", nargs = '+' , type = float)
    parser.add_argument("-num_of_iter" , "--number_of_iteration" , help = 'used for bayesopt', nargs = '?',const = 10 , type = int, default = 5)
    parser.add_argument("-init_points", "--number_of_initial_points", help = 'used for bayesopt', nargs = '?', const = 5, type = int, default = 5)
    parser.add_argument("-max_evals", "--maximum_evaluations", help = 'used for hyperopt', nargs = '?', const = 20 , type = int , default = 20)
    parser.add_argument("-cross_val", "--cross_validation", help = 'used for grid search', nargs = '?', const = 5, type = int, default = 5)
    parser.add_argument("-o", "--option" , help = "enter the type of optimization method : gridsearch, bayesopt, hyperopt")

    result = vars(parser.parse_args())

    command = result.pop('option')

    num_iter = result.pop('number_of_iteration')

    init_points = result.pop('number_of_initial_points')

    max_evals = result.pop('maximum_evaluations')

    cross_val = result.pop('cross_validation')

    param_list = {k : result[k] for k in result if result[k] != None}
    
    model_path = "/home/congnitensor/Python_projects/model_class/file_logs/model_xgb.txt"
    # model_path = "model.txt"

    obj = XgbOptimizeMethod(params = param_list, init_points = init_points , n_iter = num_iter , max_evals = max_evals, cross_val= cross_val, model_path= model_path) 

    if command == 'gridsearch':
        obj.grid_search() 

    if command == 'bayesopt':
        obj.bayes_opt()
    
    if command == 'hyperopt':
        obj.hyper_opt()






