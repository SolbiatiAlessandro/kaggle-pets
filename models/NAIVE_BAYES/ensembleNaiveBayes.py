#model specific libraries
from time import ctime
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pandas as pd
try:
    from gaussianNaiveBayes import PredictiveModel as gaussianPredictiveModel
    from multinomialNaiveBayes import PredictiveModel as multinomialPredictiveModel
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    from NAIVE_BAYES.gaussianNaiveBayes import PredictiveModel as gaussianPredictiveModel
    from NAIVE_BAYES.multinomialNaiveBayes import PredictiveModel as multinomialPredictiveModel

from functools import reduce
from operator import mul

class PredictiveModel(object):
    """
    base class for the prediction task of Adoption Prediction competition

    this is NB ensemble
    
    Naive Bayes:
    implementation for this competition is non-trivial, we can't just use scikit API out of the box

    (https://stackoverflow.com/questions/38621053/how-can-i-use-sklearn-naive-bayes-with-multiple-categorical-features)

    Now consider the case where you have a dataset consisting of several features:

        Categorical
        Bernoulli
        Normal
    Under the very assumption of using NB, these variables are independent. Consequently, you can do the following:

        Build a NB classifier for each of the categorical data separately, using your dummy variables and a multinomial NB.
        Build a NB classifier for all of the Bernoulli data at once - this is because sklearn's - - Bernoulli NB is simply a shortcut for several single-feature Bernoulli NBs.
        Same as 2 for all the normal features.

    tina huang

    By the definition of independence, the probability for an instance, is the product of the probabilities of instances by these classifiers.
    """
    
    def __init__(self, name):
        self.name = name
        self.predictions = None
        self.models = []
        print("{} [{}.__init__] initialized succesfully".format(ctime(), self.name))


    def validation(self, X, Y, mapping_sizes, method=1, verbose=False, n_folds=5):
        """
        validation method, you can choose between different validation strategies

        Args:
            mapping_sizes: list(int), (for instance mapping_size of AdoptionSpeed is 5 = len([0,1,2,3,4]), mapping_size for every categorical
            X: dataframe of (len(X.columns) - mapping_sizes) gaussian columns and (mapping_sizes) categorical columns
            Y: pandas.Series
            method number: [1,2,3]

        - 1 : Holdout (split in 2 groups) : sklearn.model_selection.ShuffleSplit
        - 2 : KFold (split in K+1 groups): sklearn.model_selection.Kfold
        - 3 : Leave-one-out (split in len(train) groups) : sklearn.model_selection.LeaveOneOut

        NOTE:
        https://www.youtube.com/watch?v=pA6uXzrDSUs&index=23&list=PLpQWTe-45nxL3bhyAJMEs90KF_gZmuqtm
        """
        if verbose: print("{} [{}.validation] start validation method {}".format(ctime(), self.name, method))
        validation_score = 0

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

            self.train(train_X, train_Y,mapping_sizes,verbose=verbose)
            predictions = self.predict(validation_X)
            score = self.evaluate(validation_Y)
            if verbose: print("{} [{}.validation] single score = {} ".format(ctime(), self.name, score))
            validation_score += score

        # the total validation score is an average of the single validation scores
        validation_score /= splitclass.get_n_splits(X)
        self.validation_score = validation_score

        if verbose: print("{} [{}.validation] validation score = {} ".format(ctime(), self.name, validation_score))
        if verbose: print("{} [{}.validation] finished validation method {}".format(ctime(), self.name, method))
        return validation_score
            
    def generate_meta_train(self, X, Y, mapping_sizes, verbose=False, n_folds=5, short=True):
        """
        generates meta features for later ensembling

        args:
            X
            Y, is needed for in-fold training
            mapping_sizes, see docs validation
            FIX: even if instead of mapping_sizes I put cat_columns like in CatBOOST I get same valid score and same meat features?
            how is this possible? mapping_sizes should be used for onehot encoding, look into this. Might be a bug in NB model

        OOF k-fold

        NOTE: see /ENSEMBLES/coursera.notes
        """

        if verbose: print("{} [{}.validation] start generate_meta mapping_sizes {}".format(ctime(), self.name, mapping_sizes))


        meta_train = pd.DataFrame({'L0':[-1 for _ in range(len(X))],
                       'L1':[-1 for _ in range(len(X))],
                       'L2':[-1 for _ in range(len(X))],
                       'L3':[-1 for _ in range(len(X))],
                       'L4':[-1 for _ in range(len(X))],
                      })

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

            self.train(train_X, train_Y, mapping_sizes, short=short)
            predictions = self.predict(validation_X, probability=True)

            assert len(predictions) == len(test_index)

            meta_train.loc[test_index] = predictions

            if verbose: print("{} [{}.validation] single fold generation for meta feature completed ".format(ctime(), self.name))

        if verbose: print("{} [{}.validation] finished meta-feat generation ".format(ctime(), self.name))

        return meta_train
        
    def generate_meta_test(self, X, Y, mapping_sizes, X_test, verbose=False, short=True):
        """
        generate meta_test feats
        same signature as generate_meta_train
        """
        if verbose: print("{} [{}.train] start generate_meta_test".format(ctime(), self.name))

        self.train(X, Y, mapping_sizes, short=short, verbose=verbose)
        meta_test = self.predict(X_test, probability=True, verbose=verbose)

        meta_test = pd.DataFrame(meta_test, columns=['L0','L1','L2','L3','L4'])

        if verbose: print("{} [{}.validation] finished meta-test generation ".format(ctime(), self.name))
        return meta_test

    def train(self, X, Y, mapping_sizes, verbose=True, short=False):
        """
        train method, feature generation is inside here, data cleaning outside
        
        Args:
            mapping_sizes: list(int), (for instance mapping_size of AdoptionSpeed is 5 = len([0,1,2,3,4]), mapping_size for every categorical
            X: dataframe of (len(X.columns) - mapping_sizes) gaussian columns and (mapping_sizes) categorical columns
            Y: pandas.Series
            short: not implemented
        """
        if verbose: print("{} [{}.train] start training".format(ctime(), self.name))


        self.gaussian_feats = X.columns[:len(X.columns) - len(mapping_sizes)]
        self.categorical_feats = X.columns[len(X.columns) - len(mapping_sizes):]

        if verbose: print("{} [{}.train] training ensemble model with gaussian_feats = {}, categorical_feats = {}".format(ctime(), self.name, self.gaussian_feats, self.categorical_feats))

        model = gaussianPredictiveModel("base-gaussianNB")
        model.train(X[self.gaussian_feats], Y, verbose=verbose)
        self.models.append(model)

        for i, categorical_feat in enumerate(self.categorical_feats):
            model = multinomialPredictiveModel("base-multinomialNB-"+categorical_feat)
            model.train(X[categorical_feat], Y, mapping_sizes[i], verbose=verbose)
            self.models.append(model)

        if verbose: print("{} [{}.train] trained succefully".format(ctime(), self.name))

        
    def predict(self, X, verbose=False, probability=False):
        """
        predict method, feature generation is inside here, data cleaning outside
        
        Args:
            X: pandas.DataFrame, shape = (, 24)
            probability: look gaussianNaiveBayes.py
        Returns:
            Y: pandas.Series
            
            
        Raise:
            .not trained
        """
        if verbose: print("{} [{}.predict] start predictions".format(ctime(), self.name))

        self.gaussian_preds = self.models[0].predict(X[self.gaussian_feats], probability=True)

        self.categorical_preds = []
        for i, categorical_feat in enumerate(self.categorical_feats):
            preds = self.models[i + 1].predict(X[categorical_feat], probability=True)
            self.categorical_preds.append(preds)

        if self.categorical_preds:
            self.categorical_preds = reduce(mul, self.categorical_preds) 
        self.predictions = self.categorical_preds * self.gaussian_preds
        if not probability: self.predictions = np.argmax(self.predictions, axis=1)

        if verbose: print("{} [{}.predict] predicted succesfully".format(ctime(), self.name))
        return self.predictions
    
    def evaluate(self, labels, verbose=False):
        """
        evaluate predictions accuracy using competition metric "Quadratic Weighted Kappa"
        more here https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
        
        Args:
            labels: truth-values, pandas.Series
        
        returns: float
        
        NOTE [Interpreting the Quadratic Weighted Kappa Metric]:
        (https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps)
        
        A weighted Kappa is a metric which is used to calculate the amount of similarity between predictions and actuals. A perfect score of  1.0 is granted when both the predictions and actuals are the same. 
        Whereas, the least possible score is -1 which is given when the predictions are furthest away from actuals. In our case, consider all actuals were 0's and all predictions were 4's. This would lead to a QWKP score of -1.
        The aim is to get as close to 1 as possible. Generally a score of 0.6+ is considered to be a really good score.
        """
        if verbose: print("{} [{}.evaluate] start evaluation".format(ctime(), self.name))
        if self.predictions is None:
            raise Exception("{} [{}.evaluate] ERROR model didn't predict, you need to call {}.predict first".format(ctime(), self.name, self.name))
            
        labels_array = np.array(labels)
        if not labels_array.shape == self.predictions.shape:
            raise Exception("{} [{}.evaluate] ERROR the shape of truth value (labels) and self.predictions is different: {} != {}, you are giving the wrong number of labels".format(ctime(), self.name, labels_array.shape, self.predictions.shape))      
            
        score = metrics.cohen_kappa_score(labels_array, self.predictions)
        
        if verbose: print("{} [{}.evaluate] evaluated succesfully".format(ctime(), self.name))
        return score
