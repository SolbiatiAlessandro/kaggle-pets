def load_data():
    import pickle as pkl
    d=pkl.load(open("../../../data/processed.pkl","rb"))

    cat_features = [0,2,3,4,5,6,7,10,11,12,13,16]
    X=d['X_train'].drop(['AdoptionSpeed'],axis=1)
    Y=d['X_train']['AdoptionSpeed']
    return X,Y,cat_features

def main():
    X,Y,cat_features = load_data()

    from catboostModel import PredictiveModel as classificationPredictiveModel

    params = {'depth':3,
            'iterations':20,
            'learning_rate':0.5,
            'l2_leaf_reg':3,
            'border_count':3,
            'thread_count':4,
    }

    model = classificationPredictiveModel("catboost_classifier_gridsearch",params)

    classifier_test_score = model.validation(X, Y, cat_features, n_folds=5, verbose=True)
    print(classifier_test_score)
    assert  classifier_test_score > 0
    # 30 secs with short True
    # 3 mins to do one fold of validation in short=False


    from catboostModelRegressor import PredictiveModel as regressionPredictiveModel

    params = {'depth':3,
            'iterations':20,
            'learning_rate':0.5,
            'l2_leaf_reg':3,
            'border_count':3,
            'thread_count':4,
    }

    model = regressionPredictiveModel("catboost_regressor_gridsearch",params)

    regressor_test_score = model.validation(X, Y, cat_features, n_folds=5, verbose=True) 
    print(regressor_test_score)
    assert regressor_test_score > 0

    # 3 mins to do one fold of validation in short=False

    from time import ctime
    from itertools import chain
    from paramsearch import paramsearch


    params = {'depth':[3,1,2,6,4,5,7,8,9,10],
              'iterations':[250,100,500,1000],
              'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
              'l2_leaf_reg':[3,1,5,10,100],
              'border_count':[32,5,10,20,50,100,200],
              'thread_count':4}


    # this function runs grid search on several parameters
    def classifier_catboost_param_tune(params):
        from catboostModel import PredictiveModel as classificationPredictiveModel
        ps = paramsearch(params)
        # search 'border_count', 'l2_leaf_reg' etc. individually 
        #   but 'iterations','learning_rate' together
        for prms in chain(ps.grid_search(['border_count']),
                          ps.grid_search(['iterations','learning_rate']),
                          ps.grid_search(['l2_leaf_reg']),
                          ps.grid_search(['depth'])):
            model = classificationPredictiveModel("catboost_classifier_gridsearch",prms)
            res = model.validation(X, Y, cat_features, n_folds=5, verbose=False)
            # save the crossvalidation result so that future iterations can reuse the best parameters
            ps.register_result(res,prms)
            print('\n\n')
            print(ctime(),res,prms)
            print("--")
            print(ps.bestscore(),ps.bestparam())
        return ps.bestparam()

    classifier_bestparams = classifier_catboost_param_tune(params)
    print(classifier_bestparams)
    import pickle as pkl
    pkl.dump(classifier_bestparams, open("classifier_bestparams.pkl","wb"))


    # this function runs grid search on several parameters
    def regressor_catboost_param_tune(params):
        from catboostModelRegressor import PredictiveModel as regressionPredictiveModel
        ps = paramsearch(params)
        # search 'border_count', 'l2_leaf_reg' etc. individually 
        #   but 'iterations','learning_rate' together
        for prms in chain(ps.grid_search(['border_count']),
                          ps.grid_search(['iterations','learning_rate']),
                          ps.grid_search(['l2_leaf_reg']),
                          ps.grid_search(['depth'])):
            model = regressionPredictiveModel("catboost_regressor_gridsearch",prms)
            res = model.validation(X, Y, cat_features, n_folds=5, verbose=False)
            # save the crossvalidation result so that future iterations can reuse the best parameters
            ps.register_result(res,prms)
            print('\n\n')
            print(ctime(),res,prms)
            print("--")
            print(ps.bestscore(),ps.bestparam())
        return ps.bestparam()


    regressor_bestparams = regressor_catboost_param_tune(params)
    print(regressor_bestparams)
    import pickle as pkl
    pkl.dump(regressor_bestparams, open("regressor_bestparams.pkl","wb"))


if __name__ == "__main__":
    main()
