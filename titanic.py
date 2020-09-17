# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:49:06 2020

@author: jenny
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



train = pd.read_csv(os.path.join("train.csv"), header=0)
test = pd.read_csv(os.path.join("test.csv"), header=0)
print(train.shape, test.shape)
plt.figure()
train.hist()


def data_preprocessing(data):
    # one hot encode Sex
    data['Sex'] = [np.where(sex=="female", 0, 1) for sex in data['Sex']]
    data['Sex'] = data['Sex'].astype('int')
    
    
    # # one hot encode Embarked
    # print(data.groupby(['Embarked']).size())
    # embark = dict(C=0, Q=1, S=2)
    # tmp = [embark[i] for i in (data['Embarked'].fillna('S'))]
    # tmp = tf.one_hot(tmp, 3).numpy().astype('int')
    # data = data.drop(labels=['Embarked'], axis=1)
    # data['Embarked_C'], data['Embarked_Q'], data['Embarked_S'] = \
    #     tmp[:,0], tmp[:,1], tmp[:,2]
    
    # after investigating encodded Embarked location and relationship with 
    # others through correlation table, the feature seems to be replaceable
    # by other features such as fare and pclass.
    # logically it tells us which is wealthier area, and age distribution
    # differences in these area. But doesn't contribute much on survival rate,
    # it's more related to where you at (cabin/pclass) on board
    data = data.drop(['Embarked'], axis=1)
    
    
    # remove prefix in Ticket
    tmp = []
    for row in data['Ticket']:
        if " " in row:
            for i in range(len(row)-1, 0, -1):
                if row[i-1]==" ":
                    tmp.append(int(row[i:]))
                    break
        
        elif "LINE" in row:
            tmp.append(int(min(data['Ticket']))-1)
        else:
            tmp.append(int(row))
    data['Ticket'] = tmp
            
    # Or drop Ticket????
    # data = data.drop(['Ticket'])
    
    
    # Drop Cabin, too many nan (687) and doesn't provide too much info
    print("instances with no Cabin value:", sum(data['Cabin'].isnull()))
    data = data.drop(['Cabin'], axis=1)
    
    
    # find which feature has high correlation relationship with Pclass
    print("instances with no Age value:", sum(data['Age'].isnull()))
    print(data.groupby('Pclass').describe()['Age'])
    print(data.corr()['Age'])
    # pd.plotting.scatter_matrix(data)
    # plt.show()
    # use mean Age of Pclass to fill up missing value
    tmp = data.groupby('Pclass').describe()['Age']['mean']
    for row in data.index[data['Age'].isnull() == True]:
        data['Age'][row] = tmp[data['Pclass'][row]]
        
    
    # fare
    # use mean Age of Pclass to fill up missing value
    tmp = data.groupby('Pclass').describe()['Fare']['mean']
    for row in data.index[data['Fare'].isnull() == True]:
        data['Fare'][row] = tmp[data['Pclass'][row]]
    
    
    # drop name
    data = data.drop(['PassengerId', 'Name'], axis=1)
    
    
    # Pclass
    print(data.groupby(['Pclass']).size())
    
    
    # SibSp and ParCh can be generalized as just family member
    data['famMem'] = data['SibSp'] + data['Parch']
    data = data.drop(['SibSp', 'Parch'], axis=1)
    
    
    print(data.corr())
    
    return data


def seperate_by_pclass(data, class_lv):
    data_class_index = data.index[data['Pclass']==class_lv]
    data_class = data.loc[data_class_index,:]
    return data_class


# Survived as label
# preprocess data
train_data = data_preprocessing(train)
train_data_1 = seperate_by_pclass(train_data, 3)
# seperate data and label
train_label = train_data_1['Survived'].copy().to_numpy()
train_data_1 = train_data_1.drop(['Survived'], axis=1).to_numpy()

train_ds = train_data_1



# normalize data: ticket(??), age, 
# or not, if uses random forest

# b) Test options and evaluation metric
# c) Spot Check Algorithms
models, names, results = [], [], []
models.append(('LR',   LogisticRegression())) 
models.append(('LDA',  LinearDiscriminantAnalysis()))
models.append(('KNN',  KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB',   GaussianNB()))
models.append(('SVC',  SVC()))

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    results.append(cross_val_score(model, train_ds, train_label, cv=kfold, 
                                   scoring='accuracy'))
    names.append(name)
    print("%s: mean=%.5f, std=%.5f" % (name, results[-1].mean(), 
                                       results[-1].std()))
"""
LR: mean=0.62860, std=0.07669
LDA: mean=0.79910, std=0.03533
KNN: mean=0.69474, std=0.03901
CART: mean=0.76553, std=0.04377
NB: mean=0.66677, std=0.04163
SVC: mean=0.61622, std=0.07049
"""
    
#try with standardized data
pipelines = []
pipelines.append(('scaledLR', 
                 Pipeline([('Scaler', StandardScaler()), 
                           ('LR', LogisticRegression())])))
pipelines.append(('scaledLDA',
                  Pipeline([('Scaler', StandardScaler()), 
                            ('LDA',  LinearDiscriminantAnalysis())])))
pipelines.append(('scaledKNN',
                 Pipeline([('Scaler', StandardScaler()), 
                           ('KNN', KNeighborsClassifier())])))
pipelines.append(('scaledCART', 
                 Pipeline([('Scaler', StandardScaler()), 
                           ('CART', DecisionTreeClassifier())])))
pipelines.append(('scaledNB', 
                 Pipeline([('Scaler', StandardScaler()), 
                           ('NB',   GaussianNB())])))
pipelines.append(('scaledSVC',
                 Pipeline([('Scaler', StandardScaler()), 
                           ('SVC', SVC())])))

for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7)
    names.append(name)
    results.append(cross_val_score(model, train_ds, train_label, cv=kfold, \
                                   scoring='accuracy'))
    print("%s: mean=%.3f, std=%.3f" % \
          (name, results[-1].mean(), results[-1].std()))
"""
scaledLR: mean=0.802, std=0.025
scaledLDA: mean=0.799, std=0.035
scaledKNN: mean=0.827, std=0.050
scaledCART: mean=0.771, std=0.027
scaledNB: mean=0.786, std=0.038
scaledSVC: mean=0.822, std=0.037
"""

# d) Compare Algorithms
fig = plt.figure()
fig.suptitle('compare scaled algorithm')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_ylabel('accuracy')
plt.show()

# 5. Improve Accuracy
# a) Algorithm Tuning
#tuning KNN
scaler = StandardScaler().fit(train_ds)
scaledX = scaler.transform(train_ds)

grid_search = dict(n_neighbors=[1,3,5,7,9,11,13,15,17,19,21])
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator=KNeighborsClassifier(), \
                    param_grid=grid_search, \
                    scoring='accuracy', cv=kfold)
grid_result = grid.fit(scaledX, train_label)

print(grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(mean, stdev, param)
"""
0.8250561797752809 {'n_neighbors': 7}
0.7790012484394506 0.05423867571692407 {'n_neighbors': 1}
0.8182397003745319 0.03205872184377888 {'n_neighbors': 3}
0.8250312109862671 0.0464966604138999 {'n_neighbors': 5}
0.8250561797752809 0.04788692336763214 {'n_neighbors': 7}
0.8216229712858928 0.0351689980146624 {'n_neighbors': 9}
0.8205118601747815 0.03932279281913453 {'n_neighbors': 11}
0.8137827715355807 0.04211869853441842 {'n_neighbors': 13}
0.8104244694132335 0.044728337994221774 {'n_neighbors': 15}
0.8115106117353308 0.040059221964869676 {'n_neighbors': 17}
0.8081273408239701 0.035376132828394115 {'n_neighbors': 19}
0.7991885143570536 0.03810711883620589 {'n_neighbors': 21}
"""

#tuning SVC
scaler = StandardScaler().fit(train_ds)
scaledX = scaler.transform(train_ds)
grid_search = dict(C=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0],
                    kernel=['linear', 'poly', 'rbf', 'sigmoid'])
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator=SVC(), param_grid=grid_search, \
                    scoring='accuracy', cv=kfold)
grid_result = grid.fit(scaledX, train_label)

print(grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(mean, stdev, param)
"""
0.828314606741573 {'C': 2.0, 'kernel': 'rbf'}
0.786729088639201 0.03937199004539473 {'C': 0.1, 'kernel': 'linear'}
0.7396629213483147 0.043429691616157166 {'C': 0.1, 'kernel': 'poly'}
0.7979900124843946 0.029646294473750848 {'C': 0.1, 'kernel': 'rbf'}
0.7766416978776529 0.037150103288694905 {'C': 0.1, 'kernel': 'sigmoid'}
0.786729088639201 0.03937199004539473 {'C': 0.3, 'kernel': 'linear'}
0.8047315855181024 0.044628748930499645 {'C': 0.3, 'kernel': 'poly'}
0.8159675405742822 0.03646795919902934 {'C': 0.3, 'kernel': 'rbf'}
0.7385518102372035 0.030412446261331345 {'C': 0.3, 'kernel': 'sigmoid'}
0.786729088639201 0.03937199004539473 {'C': 0.5, 'kernel': 'linear'}
0.8215605493133584 0.03868116127771563 {'C': 0.5, 'kernel': 'poly'}
0.8204619225967541 0.03773231192582592 {'C': 0.5, 'kernel': 'rbf'}
0.7194631710362047 0.0328573728710038 {'C': 0.5, 'kernel': 'sigmoid'}
0.786729088639201 0.03937199004539473 {'C': 0.7, 'kernel': 'linear'}
0.8215730337078652 0.03926094163013205 {'C': 0.7, 'kernel': 'poly'}
0.8226966292134831 0.03601750195270722 {'C': 0.7, 'kernel': 'rbf'}
0.7161048689138577 0.04067767485358941 {'C': 0.7, 'kernel': 'sigmoid'}
0.786729088639201 0.03937199004539473 {'C': 0.9, 'kernel': 'linear'}
0.8238077403245943 0.03787716943821078 {'C': 0.9, 'kernel': 'poly'}
0.821585518102372 0.03686201990982209 {'C': 0.9, 'kernel': 'rbf'}
0.7082521847690387 0.045869998112538155 {'C': 0.9, 'kernel': 'sigmoid'}
0.786729088639201 0.03937199004539473 {'C': 1.0, 'kernel': 'linear'}
0.8215605493133584 0.03701331587102344 {'C': 1.0, 'kernel': 'poly'}
0.821585518102372 0.03686201990982209 {'C': 1.0, 'kernel': 'rbf'}
0.704856429463171 0.03834300645130528 {'C': 1.0, 'kernel': 'sigmoid'}
0.786729088639201 0.03937199004539473 {'C': 1.3, 'kernel': 'linear'}
0.8193258426966292 0.03520567375742227 {'C': 1.3, 'kernel': 'poly'}
0.8227091136079899 0.036631453712427535 {'C': 1.3, 'kernel': 'rbf'}
0.7003995006242197 0.047728994575917755 {'C': 1.3, 'kernel': 'sigmoid'}
0.786729088639201 0.03937199004539473 {'C': 1.5, 'kernel': 'linear'}
0.8193133583021224 0.032679464028030486 {'C': 1.5, 'kernel': 'poly'}
0.8227091136079899 0.036631453712427535 {'C': 1.5, 'kernel': 'rbf'}
0.7003995006242197 0.049290495498192935 {'C': 1.5, 'kernel': 'sigmoid'}
0.786729088639201 0.03937199004539473 {'C': 1.7, 'kernel': 'linear'}
0.8182022471910113 0.03312530296088587 {'C': 1.7, 'kernel': 'poly'}
0.8249563046192259 0.03535392063154606 {'C': 1.7, 'kernel': 'rbf'}
0.69478152309613 0.05078607492654057 {'C': 1.7, 'kernel': 'sigmoid'}
0.786729088639201 0.03937199004539473 {'C': 2.0, 'kernel': 'linear'}
0.8170911360799001 0.03389463446570612 {'C': 2.0, 'kernel': 'poly'}
0.828314606741573 0.03844207600960557 {'C': 2.0, 'kernel': 'rbf'}
0.6902871410736579 0.049665576329572365 {'C': 2.0, 'kernel': 'sigmoid'}
"""



# b) Ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBR', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
ensembles.append(('scaledAB', 
                  Pipeline([('Scaler', StandardScaler()), 
                            ('AB', AdaBoostClassifier())])))
ensembles.append(('scaledGBR',
                  Pipeline([('Scaler', StandardScaler()), 
                            ('GBR', GradientBoostingClassifier())])))
ensembles.append(('scaledRF', 
                  Pipeline([('Scaler', StandardScaler()), 
                            ('RF', RandomForestClassifier())])))
ensembles.append(('scaledET',
                  Pipeline([('Scaler', StandardScaler()), 
                            ('ET', ExtraTreesClassifier())])))

results.clear()
names.clear()
for name, model in ensembles:
    kfold = KFold(n_splits=10, random_state=7)
    names.append(name)
    results.append(cross_val_score(model, train_ds, train_label, cv=kfold, \
                                    scoring='accuracy'))
    print("%s: mean=%.3f, std=%.3f" % \
          (name, results[-1].mean(), results[-1].std()))
"""
AB: mean=0.804, std=0.044
GBR: mean=0.852, std=0.036
RF: mean=0.845, std=0.036
ET: mean=0.828, std=0.039
scaledAB: mean=0.804, std=0.044
scaledGBR: mean=0.853, std=0.035
scaledRF: mean=0.844, std=0.034
scaledET: mean=0.825, std=0.037
"""

fig = plt.figure()
fig.suptitle('compare scaled ensemble algorithm')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_ylabel('accuracy')
plt.show()

#tuning ensemble GradientBoostingClassifier
scaler = StandardScaler().fit(train_ds)
scaledX = scaler.transform(train_ds)
grid_search = dict(n_estimators=[50,100,150,200,250,300,350,400],
                    criterion=['friedman_mse', 'mse', 'mae'],
                    loss=['deviance', 'exponential'])
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator=GradientBoostingClassifier(), \
                    param_grid=grid_search, scoring='accuracy', cv=kfold)
grid_result = grid.fit(scaledX, train_label)

print(grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(mean, stdev, param)
"""
0.8541198501872659 {'criterion': 'mse', 'loss': 'deviance', 'n_estimators': 350}
0.838414481897628 0.027852907978872625 {'criterion': 'friedman_mse', 'loss': 'deviance', 'n_estimators': 50}
0.8496504369538078 0.03811492185695767 {'criterion': 'friedman_mse', 'loss': 'deviance', 'n_estimators': 100}
0.8462671660424469 0.03848387051998601 {'criterion': 'friedman_mse', 'loss': 'deviance', 'n_estimators': 150}
0.8496504369538076 0.0339080406644679 {'criterion': 'friedman_mse', 'loss': 'deviance', 'n_estimators': 200}
0.8440324594257179 0.03366307886133473 {'criterion': 'friedman_mse', 'loss': 'deviance', 'n_estimators': 250}
0.8485268414481897 0.02869238987649435 {'criterion': 'friedman_mse', 'loss': 'deviance', 'n_estimators': 300}
0.8507490636704119 0.030056617309286934 {'criterion': 'friedman_mse', 'loss': 'deviance', 'n_estimators': 350}
0.8518976279650436 0.029769250090022466 {'criterion': 'friedman_mse', 'loss': 'deviance', 'n_estimators': 400}
0.8384269662921348 0.03743377774413705 {'criterion': 'friedman_mse', 'loss': 'exponential', 'n_estimators': 50}
0.8496379525593009 0.03323716295637632 {'criterion': 'friedman_mse', 'loss': 'exponential', 'n_estimators': 100}
0.8485143570536829 0.034013512311500856 {'criterion': 'friedman_mse', 'loss': 'exponential', 'n_estimators': 150}
0.844019975031211 0.03260010807972062 {'criterion': 'friedman_mse', 'loss': 'exponential', 'n_estimators': 200}
0.8485143570536829 0.030073670339944163 {'criterion': 'friedman_mse', 'loss': 'exponential', 'n_estimators': 250}
0.840649188514357 0.029511967886089653 {'criterion': 'friedman_mse', 'loss': 'exponential', 'n_estimators': 300}
0.842883895131086 0.024537552583033605 {'criterion': 'friedman_mse', 'loss': 'exponential', 'n_estimators': 350}
0.8462546816479399 0.026515822362972882 {'criterion': 'friedman_mse', 'loss': 'exponential', 'n_estimators': 400}
0.8395380774032459 0.027317957704940755 {'criterion': 'mse', 'loss': 'deviance', 'n_estimators': 50}
0.8507740324594257 0.03871668387228004 {'criterion': 'mse', 'loss': 'deviance', 'n_estimators': 100}
0.8473907615480648 0.03917710413090425 {'criterion': 'mse', 'loss': 'deviance', 'n_estimators': 150}
0.8451560549313358 0.03341128368255664 {'criterion': 'mse', 'loss': 'deviance', 'n_estimators': 200}
0.8451560549313358 0.03415863802504533 {'criterion': 'mse', 'loss': 'deviance', 'n_estimators': 250}
0.8485268414481897 0.02869238987649435 {'criterion': 'mse', 'loss': 'deviance', 'n_estimators': 300}
0.8541198501872659 0.029161218012055675 {'criterion': 'mse', 'loss': 'deviance', 'n_estimators': 350}
0.8519101123595506 0.030089711422565113 {'criterion': 'mse', 'loss': 'deviance', 'n_estimators': 400}
0.8384269662921348 0.03743377774413705 {'criterion': 'mse', 'loss': 'exponential', 'n_estimators': 50}
0.8485143570536829 0.0336402984909427 {'criterion': 'mse', 'loss': 'exponential', 'n_estimators': 100}
0.847390761548065 0.036160772298716615 {'criterion': 'mse', 'loss': 'exponential', 'n_estimators': 150}
0.842896379525593 0.03395366064200913 {'criterion': 'mse', 'loss': 'exponential', 'n_estimators': 200}
0.8496379525593009 0.028757021614465434 {'criterion': 'mse', 'loss': 'exponential', 'n_estimators': 250}
0.8417727840199749 0.02848107141545437 {'criterion': 'mse', 'loss': 'exponential', 'n_estimators': 300}
0.8440074906367041 0.02371780326697222 {'criterion': 'mse', 'loss': 'exponential', 'n_estimators': 350}
0.848501872659176 0.02611604336892014 {'criterion': 'mse', 'loss': 'exponential', 'n_estimators': 400}
0.8193008739076155 0.03809701302217757 {'criterion': 'mae', 'loss': 'deviance', 'n_estimators': 50}
0.8193008739076155 0.03809701302217757 {'criterion': 'mae', 'loss': 'deviance', 'n_estimators': 100}
0.8193008739076155 0.03809701302217757 {'criterion': 'mae', 'loss': 'deviance', 'n_estimators': 150}
0.8193008739076155 0.03809701302217757 {'criterion': 'mae', 'loss': 'deviance', 'n_estimators': 200}
0.8193008739076155 0.03809701302217757 {'criterion': 'mae', 'loss': 'deviance', 'n_estimators': 250}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'deviance', 'n_estimators': 300}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'deviance', 'n_estimators': 350}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'deviance', 'n_estimators': 400}
0.8226716604244695 0.039174239616497755 {'criterion': 'mae', 'loss': 'exponential', 'n_estimators': 50}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'exponential', 'n_estimators': 100}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'exponential', 'n_estimators': 150}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'exponential', 'n_estimators': 200}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'exponential', 'n_estimators': 250}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'exponential', 'n_estimators': 300}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'exponential', 'n_estimators': 350}
0.8204244694132334 0.03860142480237867 {'criterion': 'mae', 'loss': 'exponential', 'n_estimators': 400}
"""

#tuning ensemble ExtraTreeClassifier
scaler = StandardScaler().fit(train_ds)
scaledX = scaler.transform(train_ds)
grid_search = dict(n_estimators=[50,100,150,200,250,300,350,400],
                   criterion=['gini', 'entropy'])
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid=grid_search, \
                    scoring='accuracy', cv=kfold)
grid_result = grid.fit(scaledX, train_label)

print(grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(mean, stdev, param)
"""
0.829450686641698 {'criterion': 'entropy', 'n_estimators': 250}
0.8238077403245943 0.031308016464841035 {'criterion': 'gini', 'n_estimators': 50}
0.8282896379525593 0.030124003189818976 {'criterion': 'gini', 'n_estimators': 100}
0.8226841448189763 0.035740013137172955 {'criterion': 'gini', 'n_estimators': 150}
0.8226966292134831 0.03739328543646389 {'criterion': 'gini', 'n_estimators': 200}
0.8271660424469414 0.031019195996618298 {'criterion': 'gini', 'n_estimators': 250}
0.8237952559300874 0.032569112665730926 {'criterion': 'gini', 'n_estimators': 300}
0.8249188514357053 0.03300507476908426 {'criterion': 'gini', 'n_estimators': 350}
0.8249438202247191 0.0339831456030433 {'criterion': 'gini', 'n_estimators': 400}
0.821585518102372 0.04432598114818985 {'criterion': 'entropy', 'n_estimators': 50}
0.8282896379525593 0.03254153640210639 {'criterion': 'entropy', 'n_estimators': 100}
0.8260424469413234 0.03339757307757737 {'criterion': 'entropy', 'n_estimators': 150}
0.8260799001248438 0.03710627730375525 {'criterion': 'entropy', 'n_estimators': 200}
0.829450686641698 0.03445532075713481 {'criterion': 'entropy', 'n_estimators': 250}
0.8294257178526842 0.035344555596166546 {'criterion': 'entropy', 'n_estimators': 300}
0.8294257178526842 0.035344555596166546 {'criterion': 'entropy', 'n_estimators': 350}
0.8249313358302122 0.03215588652219541 {'criterion': 'entropy', 'n_estimators': 400}
"""



# # 6. Finalize Model
# # predict with ExtraTreesClassifier
# # b) Create standalone model on entire training dataset
# scaler = StandardScaler().fit(train_ds)
# scaledX = scaler.transform(train_ds)
# model = GradientBoostingClassifier(n_estimators=350, criterion='mse', loss='deviance')
# model.fit(scaledX, train_label)

# # a) Predictions on validation dataset
# test_data = data_preprocessing(test)
# test_ds = seperate_by_pclass(test_data, 1).to_numpy()

# scaledX_valid = scaler.transform(test_ds)
# prediction = model.predict(scaledX_valid)


# with open("prediction.csv", 'w') as file:
#     file.write("PassengerId,Survived\n")
#     for i in range(len(prediction)):
#         input_str = str(test['PassengerId'][i]) + "," + str(prediction[i]) + "\n"
#         file.write(input_str)

# # c) Save model for later use



