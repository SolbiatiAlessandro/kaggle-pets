#model specific libraries
from time import ctime
from sklearn import neighbors
from sklearn import metrics
import numpy as np
import pandas as pd

class PredictiveModel(object):
    """
    base class for the prediction task of Adoption Prediction competition

    specifically for dogs, scores 0.20 
    
    KNN-classifier, example usage inside KNN.ipynb
    """
    
    def __init__(self, name, neighbors_number=15):
        self.called = 0
        self.name = name
        self.model = neighbors.KNeighborsClassifier(neighbors_number)
        self.predictions = None
        print("{} [{}.__init__] initialized succesfully".format(ctime(), self.name))

    def validation(self, X, Y, method=1, verbose=False, n_folds=5):
        """
        validation method, you can choose between different validation strategies

        Args:
            X: pandas.DataFrame, shape = (, 24)
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

        X = self.prepare_dataset(X)

        n_splits=n_folds
        # based on method value we choose a model_selection splitclass
        from sklearn.model_selection import KFold
        splitclass = KFold(n_splits=n_splits)

        # the following 20 lines come from sklearn docs example
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
        for train_index, test_index in splitclass.split(X):

            train_X, train_Y = X.loc[train_index], Y.loc[train_index]
            validation_X, validation_Y = X.loc[test_index], Y.loc[test_index]

            assert train_X.shape[0] == train_Y.shape[0]
            assert validation_X.shape[0] == validation_Y.shape[0]

            self.train(train_X, train_Y, prepared=True)
            predictions = self.predict(validation_X, prepared=True)
            score = self.evaluate(validation_Y)
            if verbose: print("{} [{}.validation] single score = {} ".format(ctime(), self.name, score))
            validation_score += score

        # the total validation score is an average of the single validation scores
        validation_score /= splitclass.get_n_splits(X)
        self.validation_score = validation_score

        if verbose: print("{} [{}.validation] validation score = {} ".format(ctime(), self.name, validation_score))
        if verbose: print("{} [{}.validation] finished validation method {}".format(ctime(), self.name, method))
        return validation_score
            
    def generate_meta_train(self, X, Y, verbose=False, n_folds=5, short=True):
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


        meta_train = pd.DataFrame({'L0':[-1 for _ in range(len(X))],
                       'L1':[-1 for _ in range(len(X))],
                       'L2':[-1 for _ in range(len(X))],
                       'L3':[-1 for _ in range(len(X))],
                       'L4':[-1 for _ in range(len(X))],
                      })

        if n_folds < 2: n_folds = 2

        X = self.prepare_dataset(X)
            
        from sklearn.model_selection import KFold
        splitclass = KFold(n_splits=n_folds)

        # the following 20 lines come from sklearn docs example
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
        for train_index, test_index in splitclass.split(X):

            train_X, train_Y = X.loc[train_index], Y.loc[train_index]
            validation_X, validation_Y = X.loc[test_index], Y.loc[test_index]

            assert train_X.shape[0] == train_Y.shape[0]
            assert validation_X.shape[0] == validation_Y.shape[0]

            self.train(train_X, train_Y,verbose=verbose,prepared=True)
            predictions = self.predict(validation_X, probability=True,verbose=verbose, prepared=True)

            assert len(predictions) == len(test_index)

            meta_train.loc[test_index] = predictions

            if verbose: print("{} [{}.validation] single fold generation for meta feature completed ".format(ctime(), self.name))

        if verbose: print("{} [{}.validation] finished meta-feat generation ".format(ctime(), self.name))

        return meta_train

    def generate_meta_test(self, X, Y, X_test, verbose=False, short=True):
        """
        generate meta_test feats
        same signature as generate_meta_train
        """
        if verbose: print("{} [{}.train] start generate_meta_test".format(ctime(), self.name))

        self.train(X, Y, verbose=verbose, prepared=False)
        meta_test = self.predict(X_test, probability=True, verbose=verbose, prepared=False)

        meta_test = pd.DataFrame(meta_test, columns=['L0','L1','L2','L3','L4'])

        if verbose: print("{} [{}.validation] finished meta-test generation ".format(ctime(), self.name))
        return meta_test
        
    def prepare_dataset(self, X, reference_scaling=False):
        """
        drop, scale and transform data

        reference_scaling: scale proportionally to fixed measures
        """
        _X = X.copy()
        #can not be parsed by knn
        to_drop = ["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description","PetID", 'AdoptionSpeed']
        for col in to_drop:
            if col in _X.columns:
                _X.drop(col, axis=1, inplace=True)

        _X.drop(['Color3','State'],axis=1,inplace=True)
        # if this breaks means I am preparing two times
        for col in _X.columns:
            if reference_scaling:
                _X[col] = (_X[col] - self.means[col]) / (self.maxs[col] - self.mins[col])
            else:
                _X[col] = (_X[col] - _X[col].mean()) / (_X[col].max() - _X[col].min())
        _X['Sterilized'] = _X['Sterilized'] * 20
        _X['Age'] = _X['Age'] * (-90)
        _X['Breed1'] = _X['Breed1'] * 10
        _X['PhotoAmt']  = _X['PhotoAmt'] * 10
        _X['Breed2'] = _X['Breed2'] * 5

        return _X

    def train(self, X, Y, verbose=False, prepared=False, short=False):
        """
        train method, feature generation is inside here, data cleaning outside
        
        Args:
            X: pandas.DataFrame, shape = (, 24)
            Y: pandas.Series
        """
        if verbose: print("{} [{}.train] start training".format(ctime(), self.name))
        
        if not prepared:
            #this mean is for training for submission and not for validation
            self.means, self.maxs, self.mins = {}, {}, {}
            for col in X.columns:
                try:
                    self.maxs[col] = X[col].max()
                    self.mins[col] = X[col].min()
                    self.means[col] = X[col].mean()
                except:
                    #non numeric column
                    pass
            X = self.prepare_dataset(X)

        self.model.fit(X, Y)
        
        if verbose: print("{} [{}.train] trained succefully".format(ctime(), self.name))

        
    def predict(self, X, verbose=False, probability=False, prepared=False):
        """
        predict method, feature generation is inside here, data cleaning outside

        ISSUE: I should scale test data with the same scaling as in the training, now I am scaling only 'locally' and not 'globally' with train data
        
        Args:
            X: pandas.DataFrame, shape = (, 24)
            probability: (bool) predict probs
        Returns:
            Y: pandas.Series
            
        Raise:
            .not trained
        """
        if verbose: print("{} [{}.predict] start predictions".format(ctime(), self.name))

        if not prepared:
            # this is for submission not for validation
            # and should be called only one time
            self.called += 1
            assert self.called == 1
            assert self.means
            X = self.prepare_dataset(X, reference_scaling=True)

        if probability: predictions = self.model.predict_proba(X)
        else: predictions = self.model.predict(X)
        self.predictions = predictions
        
        if verbose: print("{} [{}.predict] predicted succesfully".format(ctime(), self.name))
        return predictions
    
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
            raise Exception("{} [{}.evaluate] ERROR the shape of truth value (labels) and self.predictions is different, you are giving the wrong number of labels".format(ctime(), self.name, self.name))      
            
        score = metrics.cohen_kappa_score(labels_array, self.predictions)
        
        if verbose: print("{} [{}.evaluate] evaluated succesfully".format(ctime(), self.name))
        return score
