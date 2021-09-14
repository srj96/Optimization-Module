import pandas as pd 
import numpy as np 
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from random import seed
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score


def load_data(path):
    df = pd.read_csv(path)
    return(df)

def train_test(df,target):
    label = df[target].values
    feature_col = [i for i in df.columns if i not in df[target]]
    feature = df[feature_col]
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size = 0.2 , random_state = 123)
    
    return(feature_train,feature_test,label_train,label_test)
    
def xgb_train(feature_train,label_train,feature_test):
    np.random.seed(1)
    model= XGBRegressor()
    xgb = model.fit(feature_train,label_train)
    label_predicted = xgb.predict(feature_test)
    return(label_predicted)

def objective(param):
    param = {'n_estimator': int(param['n_estimator']), 
             'max_depth': int(param['max_depth']),
             'colsample_bytree' : max(min(param['colsample_bytree'],1),0), 
             'learning_rate': max(min(param['learning_rate'],1),0)}
    
    # param = {'n_estimator': int(param['n_estimator']), 'max_depth': int(param['max_depth']), 'colsample_bytree' : max(min(param['colsample_bytree'],1),0), 'learning_rate': max(min(param['learning_rate'],1),0)}
    model = XGBRegressor(**param)
    score = cross_val_score(model, x_train, y_train).mean()
    print("Display : {}".format(score))
    return score


if __name__ == "__main__":

    file_path = '/home/congnitensor/Downloads/BostonHousing.csv'
    df = load_data(file_path)
    x_train,x_test,y_train,y_test = train_test(df,'medv')
    pbounds = {'n_estimator': (10, 30), 'max_depth': (2, 9) , 'colsample_bytree' : (0.8,1.0), 'learning_rate' : (0.3,0.5)}
    opt = BayesianOptimization(f = objective , pbounds = pbounds , random_state=1)
    opt.maximize(init_points=2,n_iter=3)
    print(opt.max)





