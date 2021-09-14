from grid_search import CogniGridSearch
from bayesopt import CogniBayesOptimize
from hoptimize import CogniHyperOptimize
import argparse
import json

class OptimizeMethod:
    
    '''Optimization base class calling other optimization technique namely Grid Search , 
    Bayes Opt and Hyper Opt.Cross Validationscore is used for grid search, 
    maximum evaluations is used for hyperopt and number of iteration
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
    parser.add_argument("-j","--config_dict",help = 'enter the json dictionary',type = json.loads)
    parser.add_argument('-o','--option',help = 'enter the option for optimization')

    result = parser.parse_args()

    dict_file = result.config_dict

    obj = OptimizeMethod(params = dict_file['params'],model_path= dict_file['model_path'],
                        init_points = dict_file['initial_points'],
                        max_evals = dict_file['maximum_eval'], 
                        cross_val = dict_file['cross_validation'],
                        n_iter = dict_file['num_of_iter']) 
    
    if result.option == 'gridsearch':
        obj.grid_search() 
    
    if result.option == 'bayesopt':
        obj.bayes_opt()
    
    if result.option == 'hyperopt':
        obj.hyper_opt() 






