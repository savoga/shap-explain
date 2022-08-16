import numpy as np
import pandas as pd

import scipy.special
import itertools

import datetime as dt

n_samples = 3000
APPROXIMATION_FACTOR = .7

df_data = pd.DataFrame(data=
                       {
                        'age':[20+np.random.randint(5) for x in range(n_samples)],
                        'location':['EU' if x%2==0 else 'US' for x in range(n_samples)],
                        'income':[80+np.random.randint(20) for x in range(n_samples)],
                        'seniority':[np.random.randint(5) for x in range(n_samples)],
                        'childs':[np.random.randint(3) for x in range(n_samples)],
                        'house':[1 if x%2==0 else 0 for x in range(n_samples)],
                        'products':[np.random.randint(10) for x in range(n_samples)]
                        })

df_data.index = ['client' + str(i+1) for i in range(n_samples)]

dict_rules = {
    'rule1':'age>40', # very important rule (strong filter)
    'rule2':'location=="EU"',
    'rule3':'income>100', # very important rule (strong filter)
    'rule4':'seniority>2',
    'rule5':'childs>1',
    'rule6':'house==1',
    'rule7':'products>7'
}

'''
According to how we defined the data, we should have something like: 
    rule6,rule2 < rule4,rule5,rule7 < rule1,rule3
'''

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
    idx_sample = list(pd.DataFrame(weights).sample(frac=APPROXIMATION_FACTOR, weights=weights).index)
    weights = weights[idx_sample]
    X = X[idx_sample]
    V = V[idx_sample]
    y = f(V)
    tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
    theta = np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))

    return theta

def prediction(client_row, activated_rules):
    rules_res = []
    if activated_rules[0]==1:
        age = client_row['age']
        rules_res.append(eval(dict_rules['rule1']))
    if activated_rules[1]==1:
        location = client_row['location']
        rules_res.append(eval(dict_rules['rule2']))
    if activated_rules[2]==1:
        income = client_row['income']
        rules_res.append(eval(dict_rules['rule3']))
    if activated_rules[3]==1:
        seniority = client_row['seniority']
        rules_res.append(eval(dict_rules['rule4']))
    if activated_rules[4]==1:
        childs = client_row['childs']
        rules_res.append(eval(dict_rules['rule5']))
    if activated_rules[5]==1:
        house = client_row['house']
        rules_res.append(eval(dict_rules['rule6']))
    if activated_rules[6]==1:
        products = client_row['products']
        rules_res.append(eval(dict_rules['rule7']))
    if False not in rules_res:
        return 1
    return 0

def predict_proba(activated_rules):
#     activated_rules = activated_rules.tolist()[0]
    label = df_data.apply(lambda x: prediction(x, activated_rules), axis=1).values
    return np.sum(label)/len(label)

def proba_combinations(activated_rules_arr):
    return np.apply_along_axis(predict_proba, 1, activated_rules_arr)

M = 7
reference = np.zeros(M)
start = dt.datetime.now()
phi = kernel_shap(proba_combinations, np.array([1,1,1,1,1,1,1]), reference, M)
print(dt.datetime.now()-start)
base_value = phi[-1]
shap_values = phi[:-1]

dict_shap = {}
for idx, rule in enumerate(dict_rules.keys()):
    dict_shap[rule] = abs(shap_values[idx])

print(dict(sorted(dict_shap.items(), key=lambda item: item[1])))