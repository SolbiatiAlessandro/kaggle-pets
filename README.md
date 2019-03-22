# kaggle-pets

This is the folder for [PetFinder.my Adoption Prediction competition](https://www.kaggle.com/c/petfinder-adoption-prediction), team "BaMoOaAl"

Team Components:
- https://www.kaggle.com/alessandrosolbiati, SolbiatiAlessandro
- https://www.kaggle.com/oanaflorescu, flores-o

We will follow along [Standford CS231-n](http://cs231n.stanford.edu/) assignments and implement them here on the competition.

------

**ROADMAP**:

Models:
- [X] KNN:
 best validation score: 0.1613
 public LB score: 0.211
- [X] NB:
 best validation score: 0.10249
 public LB score: 0.172
- [ ] implement SVM:
 best validation score:
 public LB score:
- [ ] implement NN (ResNet transfer learning)
- [X] LGBM:
 best validation score: 0.17435
 public LB score: 0.278
- [X] CATBOOST:
 best validation score: 0.20133
 public LB score:  0.349

Framework:
- [X] write standard PredictiveModel
- [X] write test
- [X] write benchmark/execution scripts ( we are using notebooks )
- [X] write docs with model performance and insight
- [X] add code coverage

Exploratory Data Analysis + Feature Engineering
- [X] Adoption Speed
- [X] Name
- [X] Age
- [ ] Breed
- [ ] Color
- [ ] Size
- [ ] Country
- [ ] Images
