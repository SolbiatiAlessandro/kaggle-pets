import pytest
import pandas as pd
import numpy as np
import os
import sys

def getXY():
    """helper to get X, Y
    
    loads data from local /data folder and add basic cleaning
    """
    # this is run from /models/KNN from pytest so the the path is correct
    os.listdir('../../data')
    assert 'out_breed.csv' in os.listdir('../../data') # this assert breaks if the data is configured uncorrectly

    breeds = pd.read_csv('../../data/out_breed.csv')
    colors = pd.read_csv('../../data/out_color.csv')
    states = pd.read_csv('../../data/out_state.csv')
    train  = pd.read_csv('../../data/out_train.csv')
    test   = pd.read_csv('../../data/out_test.csv')
    sub    = pd.read_csv('../../data/out_submission.csv')

    dogs = train[train['Type'] == 1].drop('Type',axis=1)
    cats = train[train['Type'] == 2].drop('Type',axis=1)
    X = cats.reset_index().drop('index',axis=1)
    Y = cats['AdoptionSpeed'].reset_index().drop('index',axis=1)['AdoptionSpeed']

    assert X.shape[0] == Y.shape[0]
    
    return X, Y


def test_run():
    """
    this test just runs the load-train-predict workflow

    the code is just a script of 
    85f7ec7c8c0581c347a5b8034139a9ad3a6c3352:./../kNN.ipynb
    """
    # this sys.path.append are used to import knnModel inside /models/KNN
    sys.path.append(".")
    sys.path.append("../")
    from knnCats14 import PredictiveModel

    ###########################################################
    #### this can be used as an example usage of the model ####
    ###########################################################

    X, Y = getXY() # get cats

    train_size = int(len(X)*0.8)

    # split in train and validation data
    train_X, train_Y = X[:train_size], Y[:train_size]
    validation_X, validation_Y = X[train_size:], Y[train_size:]

    assert train_X.shape[0] == train_Y.shape[0]
    assert validation_X.shape[0] == validation_Y.shape[0]

    model = PredictiveModel("KNN_run_by_pytest")
    model.train(train_X, train_Y)
    predictions = model.predict(validation_X)
    score = model.evaluate(validation_Y)

    assert score > 0 # score is less then zero means something is wrong 


def test_validation():
    """
    test cross-validation
    """
    # this sys.path.append are used to import knnModel inside /models/KNN
    sys.path.append(".")
    sys.path.append("../")
    from knnCats14 import PredictiveModel

    X, Y = getXY()
    model = PredictiveModel("KNN_run_by_pytest")
    assert model.validation(X, Y) > 0
    res1 = model.validation(X, Y, method = 1) 
    assert res1 > 0.12
    res2 = model.validation(X, Y, method = 2) 
    assert res2 > 0.14

    # if this model doesn't get more then 0.20 score means something is wrong

    # method 3 is LeaveOneOut: too costly, DEPRECATED
    # assert model.validation(X, Y, method = 3) > 0

#@pytest.mark.skip("passing")
def test_meta():
    """
    test generate_meta, replicating validation
    """
    # this sys.path.append are used to import knnModel inside /models/KNN
    sys.path.append(".")
    sys.path.append("../")
    from knnDogs20 import PredictiveModel

    X, Y = getXY()

    model = PredictiveModel("knn_by_pytest_generate_meta") 
    n_folds = 3
    score = model.validation(X, Y, n_folds=n_folds) 

    meta_train = model.generate_meta_train(X, Y, n_folds = n_folds, short=True)
    meta_train = model.generate_meta_train(X, Y, n_folds = n_folds, short=True, verbose=True)

    X = model.prepare_dataset(X)

    from sklearn.model_selection import KFold
    splitclass = KFold(n_splits=n_folds)
    for train_index, test_index in splitclass.split(X):

        meta_vals = meta_train.loc[test_index] # generated from .generate_meta
        train_X, train_Y = X.loc[train_index], Y.loc[train_index]
        validation_X, validation_Y = X.loc[test_index], Y.loc[test_index]

        assert train_X.shape[0] == train_Y.shape[0]
        assert validation_X.shape[0] == validation_Y.shape[0]

        model.train(train_X, train_Y, short=True, prepared=True)
        predictions = model.predict(validation_X, probability=True, prepared=True)

        meta_vals = meta_vals.reset_index().drop('index',axis=1)
        for i, p in enumerate(predictions):
            assert p[0] == meta_vals.loc[i, 'L0']
            assert p[1] == meta_vals.loc[i, 'L1']
            assert p[2] == meta_vals.loc[i, 'L2']
            assert p[3] == meta_vals.loc[i, 'L3']
            assert p[4] == meta_vals.loc[i, 'L4']

    """
    X_test = getXY(X_test=True)
    X_test = pd.concat([X_test[numerical_col], X_test[categorical_col]], axis=1) 
    meta_test = model.generate_meta_test(X, Y, cat_features, X_test)
    assert len(meta_test.columns) == 5
    assert len(meta_test) == len(X_test)
    """
