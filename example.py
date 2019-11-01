# Example code for some experiments

import numpy as np
from boost import *
from __init__ import * 

from sklearn.datasets import make_classification,make_regression
import pickle
import matplotlib.pyplot as plt
import os

"""Make parameters"""
params =  {
           'type_updated_coordinates': {"random":0, "ghmin": 0, 'gmin':0, 'gmax':5, 'total':0}, 
           'is_acceleration': 0, 
           'acceleration_is_line_search': 0, 
           'l2': 0,
           'type':'modified',
          'line_search_kmax':25,
          'momentum':.7}
params['linesearch_params'] = {'beta':0.9, 'sigma':1e-4,'n_iter':10}


params['l'] = 'lad'
params["T_gd"]   = 10
params["step_size"] = 1
params["is_line_search"] = True


"""Make data"""
xclass,yclass =  make_classification(n_samples=100, n_features=50, n_informative=25, n_redundant=5, 
                    n_repeated=0, n_classes=2, n_clusters_per_class=5, weights=None, 
                    flip_y=0.01, class_sep=3., hypercube=True, shift=15.0, scale=3., 
                    shuffle=True, random_state=None)

xreg, yreg  = make_regression(n_samples= 100, n_features=50, noise=0.1, random_state = None)


if(params['l'] == 'exp'):  #adaboost 
    #x, y = make_hastie_10_2(n_samples = 500, random_state = 0)
    
    x,y = xclass, yclass
    weak_learner = DecisionTreeClassifier(max_depth = 1, random_state = 1)

if(params['l'] == 'logit'):  #logitboost
    #x, y = make_hastie_10_2(n_samples = 500, random_state = 0)
    x,y = xclass, yclass

    weak_learner = DecisionTreeRegressor(max_depth = 1)

if(params['l'] == 'lad'):  # LADBoost
    x,y = xreg, yreg
    weak_learner = DecisionTreeRegressor(max_depth = 1)


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)




"""Run baselines"""

baselines = []
params['n_iter'] = 100
params['l2'] = 0
params["T_gd"]   = 10
params['is_acceleration'] = None
params['is_line_search'] = False
params['type'] = 'vanilla'
params['type_updated_coordinates'] = {"random":0, "ghmin": 0, 'gmin':0, 'gmax':0, 'total':0}
baselines.append(boost(Y_train, X_train, Y_test, X_test, params, weak_learner))

params['type'] = 'modified'
params['is_line_search'] = True
params['type_updated_coordinates'] = {"random":0, "ghmin": 0, 'gmin':0, 'gmax':0, 'total':1}
baselines.append(boost(Y_train, X_train, Y_test, X_test, params, weak_learner))



"""Run experiments"""
results = []
params['type'] = 'modified'
params['is_line_search'] = True
params['l2'] = 0.


params['is_acceleration'] = None
params['type_updated_coordinates'] = {"random":0, "ghmin": 0, 'gmin':0, 'gmax':1, 'total':0}
results.append(boost(Y_train, X_train, Y_test, X_test, params, weak_learner))

params['is_acceleration'] = 'polyak'
params['momemtum'] = 1.
results.append(boost(Y_train, X_train, Y_test, X_test, params, weak_learner))

params['is_acceleration'] = 'nesterov'
results.append(boost(Y_train, X_train, Y_test, X_test, params, weak_learner))


plotme = baselines + results
plt.figure(1,figsize=(7,3))
for r in plotme:
    plt.subplot(1,2,1)
    plt.plot(np.array(r['loss_train']))
    plt.xlabel('# weak learners')
    plt.ylabel('train loss')
    plt.subplot(1,2,2)
    plt.plot(r['loss_test'])
    plt.xlabel('# weak learners')
    plt.ylabel('test loss')
    
plt.legend(['vanilla','total',  'greedy', 'polyak','nesterov'])
plt.tight_layout()
