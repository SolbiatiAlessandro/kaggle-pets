#KNN
====

K-scores
- 0.12

This are the ideas we had about how to improve the score
- [x] research how KNN treats categorical features
- [ ] select features (age, geneder, health, fee)
- [ ] select features pairwise and draw correlation plot and visualize KNN clustering
- [x] read KNN from the book "Pattern Recognition And Machine Learning, Bishop" -> 2.5.2 Nearest-neighbour methods
- [ ] set up good cross validation
- [ ] estimate K

NOTE on validation:
====

https://www.youtube.com/watch?v=pA6uXzrDSUs&index=23&list=PLpQWTe-45nxL3bhyAJMEs90KF_gZmuqtm
- Holdout (split in 2 groups) : sklearn.model_selection.ShuffleSplit
- KFold (split in K+1 groups): sklearn.model_selection.Kfold
- Leave-one-out (split in len(train) groups) : sklearn.model_selection.LeaveOneOut

all this above methods are rowwise random splits, **we could also split data based on pet id**, this is a bit more complicated but we can comeback later on this if we find problem in validation.

NOTE on how to run code coverage:
====

pytest and coverage is installed and correctly set up
to run coverage you can use the command

pytest tests --cov=./ --cov-report html
open htmlcov/index.html
