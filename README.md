# kaggle-pets

This is the folder for [PetFinder.my Adoption Prediction competition](https://www.kaggle.com/c/petfinder-adoption-prediction), team "BaMoOaAl"

Team Components:
- https://www.kaggle.com/alessandrosolbiati, SolbiatiAlessandro
- https://www.kaggle.com/oanaflorescu, flores-o

Contributors:
- https://github.com/ferrucc-io, ferrucc-io (dramatically helpful in fixing typos)

Stage1 : (in-competition) We will follow along [Stanford CS231-n](http://cs231n.stanford.edu/) assignments and implement them here on the competition.
Stage2 : (out-competition) Upsolving of the competition

-------------------
# STAGE1 COMPETITION : ROADMAP
note: stage1 ended on 09/04/2019
<b>competition results:</b> we arrived 359th out of 1805, 18% percentile

*Models:*
- [X] KNN:
 best validation score: 0.20 + 0.14
 public LB score: 0.279 BIMODEL
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

*Framework:*
- [X] write standard PredictiveModel
- [X] write test
- [X] write benchmark/execution scripts ( we are using notebooks )
- [X] write docs with model performance and insight
- [X] add code coverage
- **[X] write implementation for Google Cloud Machine Learning Engine to run models on cloud (inside GCP)**

*Exploratory Data Analysis + Feature Engineering*
- [X] Adoption Speed
- [X] Name
- [X] Age
- [ ] Breed
- [ ] Color
- [ ] Size
- [ ] Country
- [ ] Images

---------------
# STAGE2 UPSOLVING : ROADMAP
This stage consist in finishing what we were planning to do during competition, and set up better framework/knoweldge on this known competition instead of jumping in on new competitions.

GOALS (by priority):
1. set up a consistent GCP framework/pipeline for future competition
2. explore and implement the recent autoML approaches to Kaggle
3. get insight on the specific competition

*GCP FRAMEWORK/PIPELINE*
- [X] build a running prototype running on cloud `kaggle-pets/models/TREES/XGBOOST/gcp_training/`
- [ ] write a document about GCP
- [ ] start creating a template repo to clone for every new competition

*autoML*
- [ ] explore existing solution out there and evaluate wheter to implement myself

*Competition*
- [ ] read solution and replicate them



