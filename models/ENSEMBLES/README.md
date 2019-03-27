#ENSEMBLES
====

ENSEMBLE SCORE 
(public LB in parenthesis)
- catboost non ensemble 0.201 (0.349)
- catboost + NB 0.206
- BI - catboost + NB, dogs = 0.2158, cats = 0.16821, average = 0.1920
- BI - catboost + NB + knn, dogs = 0.2206, cats = 0.177, average = 0.19879 (0.343)
- catboost + NB + knn (cats14) 0.20912 (0.342)

__________________________________________________________

NOTE on validation:
====

https://www.youtube.com/watch?v=pA6uXzrDSUs&index=23&list=PLpQWTe-45nxL3bhyAJMEs90KF_gZmuqtm
- Holdout (split in 2 groups) : sklearn.model_selection.ShuffleSplit
- KFold (split in K+1 groups): sklearn.model_selection.Kfold
- Leave-one-out (split in len(train) groups) : sklearn.model_selection.LeaveOneOut

all this above methods are rowwise random splits, **we could also split data based on pet id**, this is a bit more complicated but we can comeback later on this if we find problem in validation.

NOTE on how to run test/code coverage:
====

pytest and coverage is installed and correctly set up
to run coverage you can use the command, **you need to be inside this folder (/models/KNN)**

```
pytest tests --cov=./ --cov-report html
open htmlcov/index.html
```


