"""train.py file for GCP trainer"""
import datetime
import os
import subprocess
import pandas as pd
from time import ctime
from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

BUCKET_NAME = "kaggle-pets-dataset"
RESULTS_FOLDER = "job_results"
DATA_FOLDER = "data"
download_files = ["breed","color","state","train","test","submission"]

def download_data():
    """ 
    1. Add code to download the data from GCS (in this case, using the publicly hosted data).
    ML Engine will then be able to use the data when training your model.
    """
    os.system("mkdir -p "+DATA_FOLDER)
    for file_name in download_files:
        download_blob(BUCKET_NAME,
                    "out_"+file_name+".csv",
                    os.path.join(DATA_FOLDER, "out_"+file_name+".csv")
                )

def read_data_locally(X_test=False):
    """helper to get X, Y
    
    loads data from local /data folder and add basic cleaning

    this come from ../tests/test_run_regressor.py
    """
    data_folder = DATA_FOLDER
    assert 'out_breed.csv' in os.listdir(data_folder) # this assert breaks if the data is configured uncorrectly

    breeds = pd.read_csv(os.path.join(data_folder, 'out_breed.csv'))
    colors = pd.read_csv(os.path.join(data_folder, 'out_color.csv'))
    states = pd.read_csv(os.path.join(data_folder, 'out_state.csv'))
    train  = pd.read_csv(os.path.join(data_folder, 'out_train.csv'))
    test   = pd.read_csv(os.path.join(data_folder, 'out_test.csv'))
    sub    = pd.read_csv(os.path.join(data_folder, 'out_submission.csv'))

    X = train.drop(["AdoptionSpeed", "Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PhotoAmt","VideoAmt","PetID"], axis=1)
    Y = train['AdoptionSpeed']

    assert X.shape[0] == Y.shape[0]
    
    if X_test:
        X = test.drop(["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PhotoAmt","VideoAmt","PetID"], axis=1)
        return X
    return X, Y

def run_model(name=None, short=True):
    """
    run job on CMLE, from ../tests/run_regressor.py
    short=False for full running

    create locally two files
    ./<name>.bst : xgboost model
    ./<name>.score : validation score
    """
    from gcp_training.xgboostModelRegressor import PredictiveModel

    # [START DATA PROCESSING]
    X, Y = read_data_locally()
    string_cols = ["Unnamed: 0", "dataset_type", "Name", "RescuerID", "Description", "PetID"]
    categorical_col = ["Type","Gender","Vaccinated","Dewormed","Sterilized","Breed1","Breed2","Color1","Color2","Color3","State"]
    numerical_col = [col for col in X.columns if col not in string_cols and col not in categorical_col and col != "AdoptionSpeed"]
    mapping_sizes = [2, 2, 3, 3, 3, 307, 307, 7, 7, 7, 15]
    cat_features = [i for i in range(len(numerical_col), len(numerical_col)+len(categorical_col))]    
    X = pd.concat([X[numerical_col], X[categorical_col]], axis=1)
    # [END DATA PROCESSING]

    name = "CMLE_xgboost_regressor" if name is None else name
    model = PredictiveModel(name)

    # model validation
    score = model.validation(X, Y, categorical_col, short=short)
    os.system("mkdir -p "+RESULTS_FOLDER)
    model.model.save_model(os.path.join(RESULTS_FOLDER, name+".bst"))

    # writes validation score to file
    with open(os.path.join(RESULTS_FOLDER, name+".score"),"w") as f:
        f.write(ctime())
        f.write("\n SCORE: "+str(score))

    return score

def upload_results(name=None):
    """
    upload results to GCP bucket using APIs
    """
    os.system("mkdir -p "+RESULTS_FOLDER)
    name = "CMLE_xgboost_regressor" if name is None else name
    upload_blob(BUCKET_NAME, os.path.join(RESULTS_FOLDER, name+".bst"), os.path.join(RESULTS_FOLDER, name+".bst"))
    upload_blob(BUCKET_NAME, os.path.join(RESULTS_FOLDER, name+".score"), os.path.join(RESULTS_FOLDER, name+".score"))


if __name__ == "__main__":
    download_data()
    print(ctime()+" [train.py] starting job")
    score = run_model(short=False)
    print(ctime()+" [train.py] ended job with score "+str(score))
    upload_results()

