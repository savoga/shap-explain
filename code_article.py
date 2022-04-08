import numpy as np
import pandas as pd

import scipy.special
import itertools


# To compare the impact of *rule1* and *rule2* 
# we can't just remove these rules since *rule3* is overlapping the impact of both rules. 
# Indeed, observation 2 is filtered out by both *rule1* and *rule3*. Thus, removing *rule1* 
# or *rule2* doesn't make any difference on the result.

df_data = pd.DataFrame(data={'age':[42,39,45,51],'location':['EU','EU','US','US'],'income':[70,110,80,50]})
df_data.index = ['client_1','client_2','client_3','client_4']

dict_rules = {
    'rule1':'age>40',
    'rule2':'location=="EU"',
    'rule3':'income>60'
}

# Implementation from 
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Simple%20Kernel%20SHAP.html

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def shapley_kernel(M,s):
    if s == 0 or s == M:
        return 10000
    return (M-1)/(scipy.special.binom(M,s)*s*(M-s))

def f(X):
    np.random.seed(0)
    beta = np.random.rand(X.shape[-1])
    return np.dot(X,beta) + 10

def kernel_shap(f, x, reference, M):
    X = np.zeros((2**M,M+1))
    X[:,-1] = 1
    weights = np.zeros(2**M)
    V = np.zeros((2**M,M))
    for i in range(2**M):
        V[i,:] = reference

    for i,s in enumerate(powerset(range(M))):
        s = list(s)
        V[i,s] = x[s]
        X[i,s] = 1
        weights[i] = shapley_kernel(M,len(s))
        print('coalition {} --> weight {}'.format(s,weights[i]))
        print()
    y = f(V)
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    theta = np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y)) 
    
    tmp = np.linalg.inv(np.dot(X.T, X))
    theta = np.dot(tmp, np.dot(X.T, y))
    
    return theta

def scoring_simul(row, activated_rules):
    rules_res = []
    if activated_rules[0]==1:
        age = row['age']
        rules_res.append(eval(dict_rules['rule1']))
    if activated_rules[1]==1:
        location = row['location']
        rules_res.append(eval(dict_rules['rule2']))
    if activated_rules[2]==1:
        income = row['income']
        rules_res.append(eval(dict_rules['rule3']))
    if False not in rules_res:
        return 1
    return 0

def predict_proba(activated_rules):
#     activated_rules = activated_rules.tolist()[0]
    label = df_data.apply(lambda x: scoring_simul(x, activated_rules), axis=1).values
    return np.sum(label)/len(label)

def proba_combinations(activated_rules_arr):
    return np.apply_along_axis(predict_proba, 1, activated_rules_arr)

M = 3
reference = np.zeros(M)
phi = kernel_shap(proba_combinations, np.array([1,1,1]), reference, M)
base_value = phi[-1]
shap_values = phi[:-1]

print(shap_values)