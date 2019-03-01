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
    
    def __init__(self, name, neighbors_number=15):
        self.name = name
        self.model = neighbors.KNeighborsClassifier(neighbors_number)
        self.predictions = None
        print("{} [{}.__init__] initialized succesfully".format(ctime(), self.name))
        
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
        if not self.model:
            raise Exception("{} [{}.predict] ERROR model is not trained, you need to call {}.train first".format(ctime(), self.name, self.name))
            
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
