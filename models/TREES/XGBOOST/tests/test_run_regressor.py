import pytest
import pandas as pd
import numpy as np
import os
import sys

def getXY(X_test=False):
    """helper to get X, Y
    
    loads data from local /data folder and add basic cleaning
    """
    # this is run from /models/KNN from pytest so the the path is correct
    os.listdir('../../../data')
    assert 'out_breed.csv' in os.listdir('../../../data') # this assert breaks if the data is configured uncorrectly

    breeds = pd.read_csv('../../../data/out_breed.csv')
    colors = pd.read_csv('../../../data/out_color.csv')
    states = pd.read_csv('../../../data/out_state.csv')
    train  = pd.read_csv('../../../data/out_train.csv')
    test   = pd.read_csv('../../../data/out_test.csv')
    sub    = pd.read_csv('../../../data/out_submission.csv')

    X = train.drop(["AdoptionSpeed", "Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PhotoAmt","VideoAmt","PetID"], axis=1)
    Y = train['AdoptionSpeed']

    assert X.shape[0] == Y.shape[0]
    
    if X_test:
        X = test.drop(["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PhotoAmt","VideoAmt","PetID"], axis=1)
        return X
    return X, Y

#@pytest.mark.skip("passing")
def test_run():
    """
    this test just runs the load-train-predict workflow

    the code is just a script of 
    85f7ec7c8c0581c347a5b8034139a9ad3a6c3352../../../kNN.ipynb
    """
    # this sys.path.append are used to import knnModel inside /models/KNN
    sys.path.append(".")
    sys.path.append("../")
    from xgboostModelRegressor import PredictiveModel

    ###########################################################
    #### this can be used as an example usage of the model ####
    ###########################################################

    X, Y = getXY()

    string_cols = ["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PetID"]
    categorical_col = ["Type","Gender","Vaccinated","Dewormed","Sterilized","Breed1","Breed2","Color1","Color2","Color3","State"]
    numerical_col = [col for col in X.columns if col not in string_cols and col not in categorical_col and col != "AdoptionSpeed"]
    mapping_sizes = [2, 2, 3, 3, 3, 307, 307, 7, 7, 7, 15]
    cat_features = [i for i in range(len(numerical_col), len(numerical_col)+len(categorical_col))]

    X = pd.concat([X[numerical_col], X[categorical_col]], axis=1)

    train_size = int(len(X)*0.8)

    # split in train and validation data
    train_X, train_Y = X[:train_size], Y[:train_size]
    validation_X, validation_Y = X[train_size:], Y[train_size:]

    assert train_X.shape[0] == train_Y.shape[0]
    assert validation_X.shape[0] == validation_Y.shape[0]

    model = PredictiveModel("catboost_by_pytest")
    model.train(train_X, train_Y, cat_features)
    
    predictions = model.predict(validation_X)
    labels_predictions = model.meta_predict(train_X,train_Y,validation_X,cat_features)
    score = model.evaluate(validation_Y)

    assert score > 0 # score is less then zero means something is wrong 

    predictions = model.predict(validation_X, probability=True)
    assert len(predictions) > 0
    assert isinstance(predictions[0], float)

#@pytest.mark.skip("passing")
def test_validation():
    """
    test cross-validation
    """
    # this sys.path.append are used to import knnModel inside /models/KNN
    sys.path.append(".")
    sys.path.append("../")
    from xgboostModelRegressor import PredictiveModel

    X, Y = getXY()
    string_cols = ["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PetID"]
    categorical_col = ["Type","Gender","Vaccinated","Dewormed","Sterilized","Breed1","Breed2","Color1","Color2","Color3","State"]
    numerical_col = [col for col in X.columns if col not in string_cols and col not in categorical_col and col != "AdoptionSpeed"]
    mapping_sizes = [2, 2, 3, 3, 3, 307, 307, 7, 7, 7, 15]
    cat_features = [i for i in range(len(numerical_col), len(numerical_col)+len(categorical_col))]

    X = pd.concat([X[numerical_col], X[categorical_col]], axis=1)
    model = PredictiveModel("catboost_by_pytest")
    assert model.validation(X, Y, cat_features, n_folds=2) > 0
    assert model.validation(X, Y, cat_features, method = 1, n_folds = 2) > 0
    assert model.validation(X, Y, cat_features, method = 2, n_folds = 2) > 0
    assert model.validation(X, Y, cat_features, n_folds = 1) > 0

@pytest.mark.skip("passing")
def test_generate_meta_train():
    """
    DEPRECATED
    test generate_meta, replicating validation
    """
    # this sys.path.append are used to import knnModel inside /models/KNN
    sys.path.append(".")
    sys.path.append("../")
    from xgboostModelRegressor import PredictiveModel

    X, Y = getXY()
    string_cols = ["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PetID"]
    categorical_col = ["Type","Gender","Vaccinated","Dewormed","Sterilized","Breed1","Breed2","Color1","Color2","Color3","State"]
    numerical_col = [col for col in X.columns if col not in string_cols and col not in categorical_col and col != "AdoptionSpeed"]
    mapping_sizes = [2, 2, 3, 3, 3, 307, 307, 7, 7, 7, 15]
    cat_features = [i for i in range(len(numerical_col), len(numerical_col)+len(categorical_col))]
    X = pd.concat([X[numerical_col], X[categorical_col]], axis=1) 

    model = PredictiveModel("catboost_by_pytest_generate_meta") 
    n_folds = 3
    score = model.validation(X, Y, cat_features, n_folds=n_folds) 

    meta_train = model.generate_meta_train(X, Y, cat_features, n_folds = n_folds, short=True)
    meta_train = model.generate_meta_train(X, Y, cat_features, n_folds = n_folds, short=True, verbose=True)

    from sklearn.model_selection import KFold
    splitclass = KFold(n_splits=n_folds)
    for train_index, test_index in splitclass.split(X):

        meta_vals = meta_train.loc[test_index] # generated from .generate_meta
        train_X, train_Y = X.loc[train_index], Y.loc[train_index]
        validation_X, validation_Y = X.loc[test_index], Y.loc[test_index]

        assert train_X.shape[0] == train_Y.shape[0]
        assert validation_X.shape[0] == validation_Y.shape[0]

        model.train(train_X, train_Y, cat_features, short=True)
        predictions = model.predict(validation_X, probability=True)


        meta_vals = meta_vals.reset_index().drop('index',axis=1)
        for i, p in enumerate(predictions):
            assert p == meta_vals.iloc[i,0]

    """
    X_test = getXY(X_test=True)
    X_test = pd.concat([X_test[numerical_col], X_test[categorical_col]], axis=1) 
    meta_test = model.generate_meta_test(X, Y, cat_features, X_test)
    assert len(meta_test.columns) == 5
    assert len(meta_test) == len(X_test)
    """

#@pytest.mark.skip("passing")
def test_generate_meta_train_test():
    """
    test generate_meta, replicating validation

    generates both meta_train and meta_Test
    """
    # this sys.path.append are used to import knnModel inside /models/KNN
    sys.path.append(".")
    sys.path.append("../")
    from xgboostModelRegressor import PredictiveModel

    X, Y = getXY()
    string_cols = ["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PetID"]
    categorical_col = ["Type","Gender","Vaccinated","Dewormed","Sterilized","Breed1","Breed2","Color1","Color2","Color3","State"]
    numerical_col = [col for col in X.columns if col not in string_cols and col not in categorical_col and col != "AdoptionSpeed"]
    mapping_sizes = [2, 2, 3, 3, 3, 307, 307, 7, 7, 7, 15]
    cat_features = [i for i in range(len(numerical_col), len(numerical_col)+len(categorical_col))]
    X = pd.concat([X[numerical_col], X[categorical_col]], axis=1) 

    X_test = getXY(X_test=True)
    X_test = pd.concat([X_test[numerical_col], X_test[categorical_col]], axis=1) 
    model = PredictiveModel("catboost_by_pytest_generate_meta") 
    n_folds = 3

    # call generate_meta
    meta_train, meta_test = model.generate_meta(X, Y, X_test, cat_features, n_folds = n_folds, short=True)

    from sklearn.model_selection import StratifiedKFold
    splitclass = StratifiedKFold(n_splits=n_folds)
    for train_index, test_index in splitclass.split(X,Y):

        meta_vals = meta_train.loc[test_index] # generated from .generate_meta
        train_X, train_Y = X.loc[train_index], Y.loc[train_index]
        validation_X, validation_Y = X.loc[test_index], Y.loc[test_index]

        assert train_X.shape[0] == train_Y.shape[0]
        assert validation_X.shape[0] == validation_Y.shape[0]

        model.train(train_X, train_Y, cat_features, short=True)
        predictions = model.predict(validation_X, probability=True)


        meta_vals = meta_vals.reset_index().drop('index',axis=1)
        for i, p in enumerate(predictions):
            assert p == meta_vals.iloc[i,0]

    assert len(meta_test) == len(X_test)
