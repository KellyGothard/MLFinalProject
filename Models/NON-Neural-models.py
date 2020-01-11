#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
SEED = 0
import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import copy
import pandas as pd
from pathlib import Path
from pprint import pprint
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Constants
NB = 'naive bayes'
SVM = 'SVM'
KNN = 'kNN'
RF = 'random forest'


def make_models():
    '''
    Make a variety of model pipelines and parameter grids. Return a dict mapping from a model name
    to a tuple of (model, param_grid). The param_grid is for use with GridSearchCV

    :return: dict
    '''
    base_param_grid = {
        'vect__ngram_range': [(1, 1), (1, 2)],  # unigrams vs unigrams and bigrams
        'tfidf__use_idf': (True, False)  # term frequency only vs tf-idf.
    }

    nb =  Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultinomialNB()),
    ])
    nb_pg = {**base_param_grid}

    # SVM issues
    # Failing on internal dataset.
    # ValueError: The test_size = 4 should be greater or equal to the number of classes = 5
    # I think this means that the validation_fraction (0.1) is ~4. This makes sense, since
    # x_train is 41 observations, 5-fold cv reduces that to ~32, and 0.1 of that is ~4.
    # Fix: Make max_iter a hyperparam instead of using early stopping based on validation error.
    svm =  Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, shuffle=True,
                              max_iter=10,
                              # early_stopping=True, tol=1e-3, n_iter_no_change=5, validation_fraction=0.1
                              ))
    ])
    svm_pg = {'clf__loss': ['hinge', 'log'],
              'clf__alpha': [1e-3, 1e-4, 1e-5],
              'clf__max_iter': [10, 100, 1000],
              **base_param_grid}

    knn =  Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', KNeighborsClassifier(n_neighbors=5, weights='uniform', p=1)),
    ])
    knn_pg = {'clf__weights': ['uniform', 'distance'],
              'clf__p': [1, 2],  # manhattan distance vs euclidean distance
              'clf__n_neighbors': [1, 3, 5],
              **base_param_grid}

    rf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', RandomForestClassifier(n_estimators=10)),
    ])
    rf_pg = {'clf__n_estimators': [10, 20, 100],
             **base_param_grid}

    return {NB: (nb, nb_pg), SVM: (svm, svm_pg), KNN: (knn, knn_pg), RF: (rf, rf_pg)}


def evaluate_model(y, pred):

    print(classification_report(y, pred))
    report = classification_report(y, pred, output_dict=True)
    print('Confusion matrix: row is true class, col is predicted class')
    cm = confusion_matrix(y, pred)
    print(cm)
    return report, cm


# In[2]:


# Load Data
TRAIN_DATA = "../Data/Generated/RC_2016-10_Train.pkl"
TEST_DATA = "../Data/Generated/RC_2016-10_Test.pkl"

postsTrain = pd.read_pickle(TRAIN_DATA)
postsTest = pd.read_pickle(TEST_DATA)


# In[ ]:


kfold = 5

models = make_models()

results = {'kfold': kfold,
           'trials': []}

x_train = [' '.join(row) for row in postsTrain["tokens"].values]
y_train = postsTrain["banned"].values

x_test = [' '.join(row) for row in postsTest["tokens"].values]
y_test = postsTest["banned"].values

for model_id in models:
    result = {'model': model_id}
    model, param_grid = models[model_id]
    original_model = copy.deepcopy(model)
    print('==========================================================')
    print(f'training and evaluating {model_id}')

    print(f'dataset shapes: x_train: {len(x_train)}, x_test: {len(x_test)}, y_train: {len(y_train)}, y_test: {len(y_test)}')

    print('train set evaluation')
    model = copy.deepcopy(original_model)
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    cr_train, cm_train = evaluate_model(y_train, pred_train)
    result['classification_report_train'] = cr_train
    result['confusion_matrix_train'] = cm_train

    print(f'{kfold}-fold cross-validation model evaluation')
    model = copy.deepcopy(original_model)
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print(f'Cross-validation Accuracy: {scores.mean():.2} (+/- {scores.std() * 1.96:.2})')
    result['cv_scores_train'] = scores
    cv_pred_train = cross_val_predict(model, x_train, y_train, cv=kfold)
    cr_train_cv, cm_train_cv = evaluate_model(y_train, cv_pred_train)
    result['classification_report_train_cv'] = cr_train_cv
    result['confusion_matrix_train_cv'] = cm_train_cv

    print('Grid Search Cross Validation model evaluation: training set')
    model = copy.deepcopy(original_model)
    gs_model = GridSearchCV(model, param_grid, cv=kfold, iid=False, n_jobs=-1, refit=True)
    gs_model.fit(x_train, y_train)
    print("Grid Search Best Score:", gs_model.best_score_)
    print('Grid Search Best Params:', gs_model.best_params_)
    print('Grid Search cross-validation results:')
    pprint(gs_model.cv_results_)
    result['gs_model_cv_results_train'] = gs_model.cv_results_
    result['gs_model_best_score_train'] = gs_model.best_score_
    result['gs_model_best_params_train'] = gs_model.best_params_
    gs_pred_train = gs_model.predict(x_train)
    cr_train_gs, cm_train_gs = evaluate_model(y_train, gs_pred_train)
    result['classification_report_train_gs'] = cr_train_gs
    result['confusion_matrix_train_gs'] = cm_train_gs
    result['gs_model'] = gs_model


    # Looking at test performance leads to overfitting.
    print('Grid Search Cross validation model evaluation: test set')
    gs_pred_test = gs_model.predict(x_test)
    cr_test_gs, cm_test_gs = evaluate_model(y_test, gs_pred_test)
    result['classification_report_test_gs'] = cr_test_gs
    result['confusion_matrix_test_gs'] = cm_test_gs

    results['trials'].append(result)


# In[ ]:




