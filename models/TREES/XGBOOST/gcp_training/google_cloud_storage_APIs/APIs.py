"""API module to upload download files on the private GCP bucket for kaggle-pet competition""" 
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

if  __name__ == "__main__":
    import os
    # this is launched from ../ (where train.py resides)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./kaggle-pets-26ce7fa96dc3.json"
    download_blob("kaggle-pets-dataset","out_color.csv","cloud_colors.csv")
    upload_blob("kaggle-pets-dataset","cloud_colors.csv","out_color_copy.csv")
