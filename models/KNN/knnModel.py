#model specific libraries
from time import ctime
from sklearn import neighbors
from sklearn import metrics
import numpy as np

class PredictiveModel(object):
    """
    base class for the prediction task of Adoption Prediction competition
    
    KNN-classifier, example usage inside KNN.ipynb
    """
    
    def __init__(self, name, neighbors_number=20):
        self.name = name
        self.model = neighbors.KNeighborsClassifier(neighbors_number)
        self.predictions = None
        print("{} [{}.__init__] initialized succesfully".format(ctime(), self.name))

    def validation(self, X, Y, method=1):
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
        print("{} [{}.validation] start validation method {}".format(ctime(), self.name, method))
        validation_score = 0

        # based on method value we choose a model_selection splitclass
        if method == 1:
            from sklearn.model_selection import ShuffleSplit
            splitclass = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)

        elif method == 2:
            from sklearn.model_selection import KFold
            splitclass = KFold(n_splits=5)

        elif method == 3:
            # DEPRECATED, too costly
            from sklearn.model_selection import LeaveOneOut
            splitclass = LeaveOneOut()


        # the following 20 lines come from sklearn docs example
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
        for train_index, test_index in splitclass.split(X):

            train_X, train_Y = X.loc[train_index], Y.loc[train_index]
            validation_X, validation_Y = X.loc[test_index], Y.loc[test_index]

            assert train_X.shape[0] == train_Y.shape[0]
            assert validation_X.shape[0] == validation_Y.shape[0]

            self.train(train_X, train_Y)
            predictions = self.predict(validation_X)
            score = self.evaluate(validation_Y)
            print("{} [{}.validation] single score = {} ".format(ctime(), self.name, score))
            validation_score += score

        # the total validation score is an average of the single validation scores
        validation_score /= splitclass.get_n_splits(X)
        self.validation_score = validation_score

        print("{} [{}.validation] validation score = {} ".format(ctime(), self.name, validation_score))
        print("{} [{}.validation] finished validation method {}".format(ctime(), self.name, method))
        return validation_score
            
        
    def train(self, X, Y):
        """
        train method, feature generation is inside here, data cleaning outside
        
        Args:
            X: pandas.DataFrame, shape = (, 24)
            Y: pandas.Series
        """
        print("{} [{}.train] start training".format(ctime(), self.name))
        
        self.model.fit(X, Y)
        
        print("{} [{}.train] trained succefully".format(ctime(), self.name))

        
    def predict(self, X):
        """
        predict method, feature generation is inside here, data cleaning outside
        
        Args:
            X: pandas.DataFrame, shape = (, 24)
        Returns:
            Y: pandas.Series
            
        Raise:
            .not trained
        """
        print("{} [{}.predict] start predictions".format(ctime(), self.name))

        predictions = self.model.predict(X)
        self.predictions = predictions
        
        print("{} [{}.predict] predicted succesfully".format(ctime(), self.name))
        return predictions
    
    def evaluate(self, labels):
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
        print("{} [{}.evaluate] start evaluation".format(ctime(), self.name))
        if self.predictions is None:
            raise Exception("{} [{}.evaluate] ERROR model didn't predict, you need to call {}.predict first".format(ctime(), self.name, self.name))
            
        labels_array = np.array(labels)
        if not labels_array.shape == self.predictions.shape:
            raise Exception("{} [{}.evaluate] ERROR the shape of truth value (labels) and self.predictions is different, you are giving the wrong number of labels".format(ctime(), self.name, self.name))      
            
        score = metrics.cohen_kappa_score(labels_array, self.predictions)
        
        print("{} [{}.evaluate] evaluated succesfully".format(ctime(), self.name))
        return score
