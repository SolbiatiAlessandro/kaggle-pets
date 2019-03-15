IDEAS FOR NEW FEATURES
======================

[Got some ideas by looking at the website](https://www.petfinder.my/listings.php?sorttype=1&sortadopt=0)

- [ ] look likes the owner track record of adoption is pretty relevant, if they know how to make nice insertions. One features could be #adopted_pet_by_owner
- [ ] the number of pics/video
- [ ] the quality of the pics, especially profile pic
- [ ] they have this thing called sponsored listing, if we are able to find out if is sponsored that is a high signal
- [ ] so there is this scammers, that offer really interesting animals for low prices. If we are able to tell the model that those fake insertion are considered really attractive by people might be a signal -> https://www.petfinder.my/scams.htm
- [ ] you can share on facebook/on your blog the insertion, might be interesting
- [ ] check if there are comments under the pic
- [ ] cuteness meter, they have a model on the website that given a pic tells you how cute it is! we could reverse engineer that
- [ ] NLP features (suggested by my mom), stuff like "character", "if he was from the street or born in the house"

(ideas from brainstorming session on 02 March)
- [ ] does the name has number inside? if yes is not cute
- [ ] is all about the pictures -> size of the face (have a CNN that tells you that)
- [ ] video length
- [ ] brightness of the pic (or some sort of CNN that tells you how good is an image)


(one week later ideas, after having explored more the dataset)
- encoders for breed (int mapping) -> n-dimensional vectors
- encoders for state (int mapping) -> 2d state representation
- similarity of pets from images 
- knn features to be later fed into XGBoost
- mean encoding
- features interaction encodings

FEATURES
========

[NAME_features]
- FEAT #00 : (binary) the pet has a name
- FEAT #01 : (binary) the pet has a number in the name
- FEAT #02 : (binary) the pet has a coded name (e.g. BA21)
- FEAT #03 : (binary) is the name frequent (appeared more than 10 times?)
- FEAT #05: is cat and is cat's age less then 5 months
