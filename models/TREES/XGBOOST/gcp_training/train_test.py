import os
import sys
sys.path.append(".") # it depends where you launch tests
sys.path.append("../")
import train
import pandas as pd
import pytest

RESULTS_FOLDER = train.RESULTS_FOLDER
DATA_FOLDER = train.DATA_FOLDER
download_files = train.download_files

@pytest.mark.skip("passing") 
def test_download_data():
    """test gcs download data"""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./google_cloud_storage_APIs/kaggle-pets-dff39db5da9c.json"
    os.system("rm -r "+DATA_FOLDER)
    train.download_data()

    for file_name in download_files:
        assert os.path.isfile(
                os.path.join(DATA_FOLDER, "out_"+file_name+".csv")
                )

@pytest.mark.skip("passing") 
def test_read_data_locally():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./google_cloud_storage_APIs/kaggle-pets-dff39db5da9c.json"
    os.system("rm -r "+DATA_FOLDER)
    train.download_data()
    X, Y = train.read_data_locally()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.Series)

def test_run_job():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./google_cloud_storage_APIs/kaggle-pets-dff39db5da9c.json" # this should not be needed on the cloud
    os.system("rm -r "+DATA_FOLDER)
    os.system("rm -r "+RESULTS_FOLDER)
    train.download_data()

    name = "test_xgboost"
    train.run_model(name)

    assert os.path.isfile(
            os.path.join(RESULTS_FOLDER, name+".bst")
            )
    assert os.path.isfile(
            os.path.join(RESULTS_FOLDER, name+".score")
            )
