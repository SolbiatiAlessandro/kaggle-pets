PROJECT_ID=kaggle-pets
BUCKET_ID=kaggle-pets-dataset
REGION=US
TRAINER_PACKAGE_PATH=./gcp_training/
MAIN_TRAINER_MODULE=./gcp_training/train
JOB_DIR=gs://kaggle-pets-dataset/
RUNTIME_VERSION=1.9
PYTHON_VERSION=3.5
gcloud config set project kaggle-pets
gcloud ml-engine jobs submit training xgboost_regressor_$(date +"%Y%m%d_%H%M%S") \
  --job-dir $JOB_DIR \
  --package-path $TRAINER_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=$PYTHON_VERSION \
  --scale-tier BASIC
