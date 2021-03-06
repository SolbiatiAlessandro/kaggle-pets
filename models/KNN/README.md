#KNN
====

K-scores
- 0.12
- 0.13190454691363668 (k = 20)
- 0.09 [I dropped all categorical features, and score decreased]
- 0.002 [I just used pairs of features]
- 0.135 (k=25, add photo and video)
- 0.10 scaled
- 0.1613 k = 100, only dogs (for cats 0.09)
- 0.20 DOGS standardized, 0.14 cats standardized
- BIMODEL on LB is 0.27

__________________________________________________________

This are the ideas we had about how to improve the score
- [x] research how KNN treats categorical features
- [X] select features (age, geneder, health, fee)
- [X] select features pairwise and draw correlation plot and visualize KNN clustering
- [x] read KNN from the book "Pattern Recognition And Machine Learning, Bishop" -> 2.5.2 Nearest-neighbour methods
- [X] set up good cross validation

looks like selecting features only decreases score, looks like there is a lot of signal in all the features (is pretty strange since most of them are mapped)

- [ ] estimate K (run cross validation and get best K) **easy**
- [ ] play around with scikit paratmeters of KNN, using different distance and algoss
- [ ] we should get some visualization tool to see how KNN is splitting
- [ ] normalize input features **easy**
- [ ] dimensionality reduction PCA

if we wanted to really work on the KNN we could think of designing some distance function that takes into account the mapping, for instance betweens similar colors and between similar breeds. This might be useful later, we can think about distance functions.

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


