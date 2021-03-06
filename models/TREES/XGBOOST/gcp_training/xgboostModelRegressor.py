#mod specific libraries
from time import ctime
import xgboost as xgb
import sklearn.metrics as metrics

import scipy as sp
import pandas as pd
import numpy as np

class PredictiveModel(object):
    """
    base class for the prediction task of Adoption Prediction competition

    this is catboost!

    https://www.coursera.org/learn/competitive-data-science/notebook/4daDt/catboost-notebook 
    """
    
    def __init__(self, name, params = None):
        self.name = name
        self.model = None
        self.predictions = None
        self.train_results = {}
        if params is not None:
            self.params = params
        else:
            self.params = None
        print("{} [{}.__init__] initialized succesfully".format(ctime(), self.name))

    def validation(self, X, Y, cat_features, method=1, verbose=False, n_folds = 5, short=True):
        """
        validation method, you can choose between different validation strategies

        Args:
            X: pandas.DataFrame, shape = (, 24)
            Y: pandas.Series
            method number: [1,2,3] # deprecated for ensemble
            cat_features: [9,10,11] see .train docstring
            n_folds: > 2


        always using k-fold, if n_folds is 1 it is automatically put to 2

        NOTE:
        https://www.youtube.com/watch?v=pA6uXzrDSUs&index=23&list=PLpQWTe-45nxL3bhyAJMEs90KF_gZmuqtm
        """
        if verbose: print("{} [{}.validation] start validation method {}".format(ctime(), self.name, method))
        validation_score = 0

        if n_folds < 2: n_folds = 2

        from sklearn.model_selection import StratifiedKFold
        splitclass = StratifiedKFold(n_splits=n_folds)

        # the following 20 lines come from sklearn docs example
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
        for train_index, test_index in splitclass.split(X, Y):

            train_X, train_Y = X.loc[train_index], Y.loc[train_index]
            validation_X, validation_Y = X.loc[test_index], Y.loc[test_index]

            assert train_X.shape[0] == train_Y.shape[0]
            assert validation_X.shape[0] == validation_Y.shape[0]

            train_X.reset_index(drop=True,inplace=True)
            train_Y.reset_index(drop=True,inplace=True)
            validation_X.reset_index(drop=True,inplace=True)
            validation_Y.reset_index(drop=True,inplace=True)
            self.meta_predict(train_X,train_Y,validation_X,cat_features,short=short)
            score = self.evaluate(validation_Y)

            if verbose: print("{} [{}.validation] single score = {} ".format(ctime(), self.name, score))
            validation_score += score

        # the total validation score is an average of the single validation scores
        validation_score /= splitclass.get_n_splits(X)
        self.validation_score = validation_score

        if verbose: print("{} [{}.validation] validation score = {} ".format(ctime(), self.name, validation_score))
        if verbose: print("{} [{}.validation] finished validation method {}".format(ctime(), self.name, method))

        return validation_score

    
    def generate_meta_train(self, X, Y, cat_features, verbose=False, n_folds=5, short=True):
        """
        generates meta features for later ensembling

        args:
            X
            Y, is needed for in-fold training
            cat_features: [9,10,11] see .train docstring

        OOF k-fold

        NOTE: see /ENSEMBLES/coursera.notes
        """

        if verbose: print("{} [{}.validation] start generate_meta short={}".format(ctime(), self.name, short))


        meta_train = pd.DataFrame({'{}_regressor'.format(self.name):[-1 for _ in range(len(X))] })

        if n_folds < 2: n_folds = 2
            
        from sklearn.model_selection import KFold
        splitclass = KFold(n_splits=n_folds)

        # the following 20 lines come from sklearn docs example
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
        for train_index, test_index in splitclass.split(X):

            train_X, train_Y = X.loc[train_index], Y.loc[train_index]
            validation_X, validation_Y = X.loc[test_index], Y.loc[test_index]

            assert train_X.shape[0] == train_Y.shape[0]
            assert validation_X.shape[0] == validation_Y.shape[0]

            self.train(train_X, train_Y, cat_features, short=short,verbose=verbose)
            predictions = self.predict(validation_X, probability=True,verbose=verbose)

            assert len(predictions) == len(test_index)

            meta_train.iloc[test_index,0] = predictions

            if verbose: print("{} [{}.validation] single fold generation for meta feature completed ".format(ctime(), self.name))

        if verbose: print("{} [{}.validation] finished meta-feat generation ".format(ctime(), self.name))

        return meta_train
        
    def generate_meta_test(self, X, Y, cat_features, X_test, verbose=False, short=True):
        """
        DEPRECATED, returned by generate_meta_train

        generate meta_test feats
        same signature as generate_meta_train
        """
        if verbose: print("{} [{}.train] start generate_meta_test".format(ctime(), self.name))

        self.train(X, Y, cat_features, short=short, verbose=verbose)
        meta_test = self.predict(X_test, probability=True, verbose=verbose)

        meta_test = pd.DataFrame(meta_test, columns=['L0','L1','L2','L3','L4'])

        if verbose: print("{} [{}.validation] finished meta-test generation ".format(ctime(), self.name))
        return meta_test

    def generate_meta(self, X, Y, X_test, cat_features, verbose=False, n_folds=5, short=True):
        """
        generates meta features for later ensembling

        args:
            X
            Y, is needed for in-fold training
            cat_features: [9,10,11] see .train docstring

        OOF k-fold

        returns:
        meta_train, meta_test (predictions)

        meta_test is actually a average (bagged) version of 5 different trained models

        NOTE: see /ENSEMBLES/coursera.notes
        """

        if verbose: print("{} [{}.validation] start generate_meta short={}".format(ctime(), self.name, short))

        if n_folds < 2: n_folds = 2

        meta_train =pd.DataFrame({'{}_regressor'.format(self.name):[-1 for _ in range(len(X))] })
        meta_test = np.zeros((X_test.shape[0], n_folds))

        from sklearn.model_selection import StratifiedKFold
        splitclass = StratifiedKFold(n_splits=n_folds)

        fold = 0
        for train_index, test_index in splitclass.split(X, Y):

            train_X, train_Y = X.loc[train_index], Y.loc[train_index]
            validation_X, validation_Y = X.loc[test_index], Y.loc[test_index]

            assert train_X.shape[0] == train_Y.shape[0]
            assert validation_X.shape[0] == validation_Y.shape[0]

            self.train(train_X, train_Y, cat_features, short=short,verbose=verbose)
            val_predictions = self.predict(validation_X, probability=True,verbose=verbose)
            test_predictions = self.predict(X_test, probability=True,verbose=verbose)

            assert len(val_predictions) == len(test_index)
            assert len(test_predictions) == len(X_test)

            meta_train.iloc[test_index,0] = val_predictions
            meta_test[:, fold] = test_predictions
            fold += 1

            if verbose: print("{} [{}.validation] single fold generation for meta feature completed ".format(ctime(), self.name))

        # here the meta-predictions get averaged
        meta_test = meta_test.mean(axis=1) 

        if verbose: print("{} [{}.validation] finished meta-feat generation ".format(ctime(), self.name))

        return meta_train, meta_test

    def train(self, X, Y, cat_features, verbose=False, split_len=0.8, short=True):
        """
        train method, feature generation is inside here, data cleaning outside
        
        Args:
            X: pandas.DataFrame
            Y: pandas.Series
            cat_features: [6,7,8,9] index of cat columns, all numerical and then all categorical
        """
        if verbose: print("{} [{}.train] start training".format(ctime(), self.name))

        split = int(len(X)*split_len)
        X_train, Y_train = X[:split], Y[:split]
        X_val, Y_val = X[split:], Y[split:]
        # FIRST ALL NUMERICAL, then ALL CATEGORICAL

        verbose_eval = 1000
        early_stop = 500

        xgb_params = {
            'eval_metric': 'rmse',
            'seed': 1337,
            'eta': 0.0123,
            'subsample': 0.8,
            'colsample_bytree': 0.85,
            'silent': 1,
        }

        self.features = X.columns

        d_train = xgb.DMatrix(data=X_train, label=Y_train, feature_names=self.features)
        d_valid = xgb.DMatrix(data=X_val, label=Y_val, feature_names=self.features)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        iterations = 500 if not short else 10
        if self.params and self.params.get('iterations') is not None:
            iterations = self.params['iterations']

        self.model = xgb.train(dtrain=d_train,
                num_boost_round=iterations,
                evals=watchlist,
                early_stopping_rounds=early_stop, 
                verbose_eval=verbose_eval, params=xgb_params)

        if verbose: print("{} [{}.train] trained succefully".format(ctime(), self.name))

    def predict(self, X, verbose=False, probability=False):
        """
                verbose_eval=verbose)
        predict method, feature generation is inside here, data cleaning outside
        
        Args:
            X: pandas.DataFrame, shape = (, 24)
            probability: True to predict probability estimates for test vector X, False to predict classes
        Returns:
            Y: pandas.Series (if probability False)
            Y: pandas.DataFrame, shape = (, 5) a probability for every class
            e.g. [0.05986134, 0.46925879, 0.23683668, 0.22380147, 0.01024172]
            
        Raise:
            .not trained
        """
        if verbose: print("{} [{}.predict] start predictions".format(ctime(), self.name))

        predictions = self.model.predict(xgb.DMatrix(X, feature_names=self.features), ntree_limit=self.model.best_ntree_limit)
        self.predictions = predictions
        
        if verbose: print("{} [{}.predict] predicted succesfully".format(ctime(), self.name))
        return predictions

    def meta_predict(self,X,Y,X_test,cat_features,verbose=False,short=True):
        """
        predict labels, this is the last layer of the meta model
        """
        meta_train, meta_test = self.generate_meta(X,Y,X_test,cat_features,verbose=verbose,short=short)
        meta_train =  np.array(meta_train.iloc[:,0])

        import sys
        sys.path.append("../../")
        from gcp_training.rounder import OptimizedRounder
        rounder = OptimizedRounder()
        rounder.fit(meta_train, Y)
        coefficients = rounder.coefficients()
        self.labels_predictions = rounder.predict(meta_test, coefficients)
        return self.labels_predictions

    def visualize(self, verbose=False):
        """
        if you call it from jupyter use
        %matplotlib inline

        visualize training results
        """
        if verbose: print("{} [{}.visualize] start visualizing".format(ctime(), self.name))

        assert self.model is not None

        from matplotlib import pyplot as plt
        plt.barh(self.model.feature_names_, self.model.feature_importances_)

        if verbose: print("{} [{}.visualzed] visualized succesfully".format(ctime(), self.name))

    def evaluate(self, labels, verbose=False):
        """
        evaluate predictions accuracy using competition metric "Quadratic Weighted Kappa"
        more here https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
        
        Args:
            labels: truth-values, pandas.Series
        
        returns: float
        
        NOTE [Interpreting the Quadra Weighted Kappa Metric]:
        (https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps)
        
        A weighted Kappa is a metric which is used to calculate the amount of similarity between predictions and actuals. A perfect score of  1.0 is granted when both the predictions and actuals are the same. 
        Whereas, the least possible score is -1 which is given when the predictions are furthest away from actuals. In our case, consider all actuals were 0's and all predictions were 4's. This would lead to a QWKP score of -1.
        The aim is to get as close to 1 as possible. Generally a score of 0.6+ is considered to be a really good score.
        """
        if verbose: print("{} [{}.evaluate] start evaluation".format(ctime(), self.name))
        if self.labels_predictions is None:
            raise Exception("{} [{}.evaluate] ERROR model didn't predict, you need to call {}.predict first".format(ctime(), self.name, self.name))
            
        labels_array = np.array(labels)
        if not labels_array.shape[0] == self.labels_predictions.shape[0]:
            raise Exception("{} [{}.evaluate] ERROR the shape of truth value (labels) and self.predictions is different, you are giving the wrong number of labels: {}, {}".format(ctime(), self.name, labels_array.shape, self.labels_predictions.shape))      
            
        score = metrics.cohen_kappa_score(labels_array, self.labels_predictions)
        
        if verbose: print("{} [{}.evaluate] evaluated succesfully".format(ctime(), self.name))
        return score
