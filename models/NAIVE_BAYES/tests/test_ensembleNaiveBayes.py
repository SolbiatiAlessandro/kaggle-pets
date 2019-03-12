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

    X = train.drop(["AdoptionSpeed", "Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PhotoAmt","VideoAmt","PetID"], axis=1)
    Y = train['AdoptionSpeed']

    assert X.shape[0] == Y.shape[0]
    
    return X, Y


def test_run():
    """
    this test just runs the load-train-predict workflow

    the code is just a script of 
    85f7ec7c8c0581c347a5b8034139a9ad3a6c3352:./../kNN.ipynb
    """
    # this sys.path.append are used to import gaussianNaiveBayes inside /models/KNN
    sys.path.append(".")
    sys.path.append("../")
    from ensembleNaiveBayes import PredictiveModel

    ###########################################################
    #### this can be used as an example usage of the model ####
    ###########################################################

    X, Y = getXY()

    string_cols = ["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PhotoAmt","VideoAmt","PetID"]
    categorical_col = ["Type","Gender","Vaccinated","Dewormed","Sterilized","Breed1","Breed2","Color1","Color2","Color3","State"]
    mapping_sizes = [2, 2, 3, 3, 3, 307, 307, 7, 7, 7, 15] 
    numerical_col = [col for col in X.columns if col not in string_cols and col not in categorical_col and col != "AdoptionSpeed"]

    X = pd.concat([X[numerical_col], X[categorical_col]], axis=1)

    train_size = int(len(X)*0.8)

    # split in train and validation data
    train_X, train_Y = X[:train_size], Y[:train_size]
    validation_X, validation_Y = X[train_size:], Y[train_size:]

    assert train_X.shape[0] == train_Y.shape[0]
    assert validation_X.shape[0] == validation_Y.shape[0]

    model = PredictiveModel("eNB_run_by_pytest")
    model.train(train_X, train_Y, mapping_sizes)

    assert model.models
    assert model.models[0].name == "base-gaussianNB"
    for i, categorical_feat in enumerate(categorical_col):
        # i + 1 is because the first is gaussian
        assert model.models[i + 1].name == "base-multinomialNB-"+categorical_feat
    
    predictions = model.predict(validation_X, probability = True)

    assert model.gaussian_preds is not None
    assert model.categorical_preds is not None

    assert len(model.categorical_preds[0]) == 5
    assert 1 - 1e6 < sum(model.categorical_preds[0]) < 1 + 1e6

    assert len(model.gaussian_preds[0]) == 5
    assert 1 - 1e6 < sum(model.gaussian_preds[0]) < 1 + 1e6

    assert predictions is not None
    assert predictions[0] is not None
    assert len(predictions[0]) == 5

    predictions = model.predict(validation_X)
    assert predictions is not None
    assert isinstance( predictions[0], np.int64 )

    score = model.evaluate(validation_Y)
    assert score > 0


def test_validation():
    """
    test cross-validation
    """
    # this sys.path.append are used to import gaussianNaiveBayes inside /models/KNN
    sys.path.append(".")
    sys.path.append("../")
    from ensembleNaiveBayes import PredictiveModel

    X, Y = getXY()

    string_cols = ["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PhotoAmt","VideoAmt","PetID"]
    categorical_col = ["Type","Gender","Vaccinated","Dewormed","Sterilized","Breed1","Breed2","Color1","Color2","Color3","State"]
    mapping_sizes = [2, 2, 3, 3, 3, 307, 307, 7, 7, 7, 15] 
    numerical_col = [col for col in X.columns if col not in string_cols and col not in categorical_col and col != "AdoptionSpeed"]

    X = pd.concat([X[numerical_col], X[categorical_col]], axis=1)

    model = PredictiveModel("ensemble_run_by_pytest")
    assert model.validation(X, Y, mapping_sizes) > 0
    assert model.validation(X, Y, mapping_sizes, method = 1) > 0
    assert model.validation(X, Y, mapping_sizes, method = 2) > 0

    # method 3 is LeaveOneOut: too costly, DEPRECATED
    # assert model.validation(X, Y, method = 3) > 0
