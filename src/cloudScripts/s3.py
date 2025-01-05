import boto3
import os
from botocore.exceptions import NoCredentialsError


def upload_to_s3(bucket_name, bucket_dir_path, local_dir_path):
    s3 = boto3.client("s3")
    for root, dirs, files in os.walk(local_dir_path):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_path = os.path.join(
                bucket_dir_path, local_path[len(local_dir_path) + 1 :]
            )
            try:
                s3.upload_file(local_path, bucket_name, s3_path)
            except FileNotFoundError:
                pass  # Handle the exception as needed


def download_from_s3(bucket_name, bucket_dir_path, final_dir_path, files_ext):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    objs = list(bucket.objects.filter(Prefix=bucket_dir_path))

    for obj in objs:
        file_name = os.path.basename(obj.key)
        _, file_extension = os.path.splitext(file_name)
        if file_extension == files_ext:
            out_path = os.path.join(final_dir_path, file_name)
            bucket.download_file(obj.key, out_path)
