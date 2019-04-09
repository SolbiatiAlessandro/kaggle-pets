Google CMLE (Cloud Machine Learning Engine) configuration
=========================

**folder content**:

__init__.py
google_cloud_storage_APIs
 this are APIs for upload/download from bucket	
train.py 
 this is where the job for the CLME is
 what it does is
1. Download data from bucket
2. Run validation and train model
3. Upload to bucket results

xgboostModelRegressor.py
 this is the model imported from ../

./google_cloud_storage_APIs:
APIs.py
__init__.py
kaggle-pets-dff39db5da9c.json
 you need credential access to access bucket


USAGE
=======

There is a notebook with a detailed description on how to run this job on GCP CMLE inside ../submit_gcp_job.ipynb

You can also just run the command ../submit_gcp_job.sh

