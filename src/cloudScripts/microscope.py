import models as mm
import customDatasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
import json
import os
import s3
import torch
import time
import sys

# Check args for downloading images
if len(sys.argv) != 3:
    print("Usage: python script.py <y/n> <y/n>")
    sys.exit(1)

download_microscope_images = False
download_camera_images = False

if sys.argv[1] == "y":
    download_microscope_images = True

if sys.argv[2] == "y":
    download_camera_images = True


# Create dirs structure
microscope_images_local_dir_path = "/home/ec2-user/data/images/microscope"
camera_images_local_dir_path = "/home/ec2-user/data/images/camera"
models_metadata_local_dir_path = "/home/ec2-user/metadata"
code_local_dir_path = "/home/ec2-user/code"  # This dir is created manually

print("Creating dir structure")
os.makedirs(microscope_images_local_dir_path, exist_ok=True)
os.makedirs(camera_images_local_dir_path, exist_ok=True)
os.makedirs(models_metadata_local_dir_path, exist_ok=True)

# Download data
bucket_name = "mcgill-thesis"

# source images bucket paths
bucket_dir_path_micro_imgs = "data/images/microscope"
bucket_dir_path_camera_imgs = "data/images/camera"

# source images local paths
final_dir_path_micro_imgs = microscope_images_local_dir_path
final_dir_path_camera_imgs = camera_images_local_dir_path

if download_microscope_images:
    print("Downloading Microscope images from S3")
    s3.download_from_s3(
        bucket_name, bucket_dir_path_micro_imgs, final_dir_path_micro_imgs, ".jpg"
    )

if download_camera_images:
    print("Downloading Camera images from S3")
    s3.download_from_s3(
        bucket_name, bucket_dir_path_camera_imgs, final_dir_path_camera_imgs, ".jpg"
    )

# General training varibales
sets_json_file_path = "/home/ec2-user/code/80-20-indices.json"
labels_file_path = "/home/ec2-user/code/revisedLabelsV2.csv"
label_type = "om-regression"
save_model_path = models_metadata_local_dir_path
save_model_meta_path = models_metadata_local_dir_path

# Define model and training configurations
imgs_dir_path = microscope_images_local_dir_path

img_roi = 1920
img_divisions_n = 2

img_channels = 3
height_resize = 224
width_resize = 224
rotations = [0, 90, 180, 270]

training_jobs = [
    {
        "model_class": "ConvNext-All",
        "model_name": "Microscope-2Subimages-NonSoftmax",
        "model_structure": {"model_type": "tiny", "pre_trained": True},
        "training_info": {
            "n_epochs": 30,
            "save_from_epoch": 25,
            "batch_size": 32,
            "optimizer_name": "Adam",
            "optimizer_params": {"lr": 0.0001, "weight_decay": 0},
            "cross_entropy_loss_weights": None,
            "train_only_last_layer": False,
        },
        "data_info": {
            "sets_json_file_path": sets_json_file_path,
            "images_dir_path": imgs_dir_path,
            "labels_file_path": labels_file_path,
            "label_type": label_type,  # This is the only field that is used in the model trainer
            "img_roi": img_roi,
            "img_divisions_n": img_divisions_n,
            "img_channels": img_channels,
            "height_resize": height_resize,
            "width_resize": width_resize,
            "rotations": rotations,
        },
        "saving_info": {
            "save_model": 1,
            "save_model_path": save_model_path,
            "save_model_meta_path": save_model_meta_path,
        },
    }
]

# Get sets idxs
with open(sets_json_file_path, "r") as json_file:
    set_dict = json.load(json_file)

train_samples_idxs = set_dict["train"]

print(f"Train images: {len(train_samples_idxs)}")

# Build cropboxes
sub_img_dim = img_roi // img_divisions_n
cropboxes = customDatasets.createCropBoxes(img_divisions_n, sub_img_dim)

# Transforms
input_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Build DataSets
training_dataset = customDatasets.OneImageCropboxRotationDataset(
    imgs_dir_path,
    labels_file_path,
    label_type,
    train_samples_idxs,
    cropboxes,
    img_channels,
    height_resize,
    width_resize,
    rotations_values=rotations,
    transform=input_transform,
)

training_dataset_n = training_dataset.__len__()

print(
    "Train set has {} samples with shape: {}".format(
        training_dataset_n, training_dataset[0][0].shape
    )
)
print(
    "Train[0] min,max: {}, {}".format(
        torch.min(training_dataset[0][0]), torch.max(training_dataset[0][0])
    )
)

# Save jobs info
jobs_time_tag = int(time.time())
jobs_info_path = save_model_meta_path + "/jobs_{}.json".format(jobs_time_tag)
with open(jobs_info_path, "w") as json_file:
    json.dump(training_jobs, json_file, indent=4)


for job in training_jobs:
    # Create DataLoaders (shuffle for training, not for validation)
    training_loader = DataLoader(
        training_dataset,
        batch_size=job["training_info"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Build and train model
    model_trainer = mm.ConvnextTrainer(
        job["model_class"],
        job["model_name"],
        job["model_structure"],
        job["training_info"],
        job["data_info"],
        job["saving_info"],
    )
    model_trainer.start_training(
        training_loader,
        training_dataset_n,
        epoch_logs_n=1,
    )


# Upload data
local_dir_path = "/home/ec2-user/metadata"
bucket_dir_path = "metadata-micro"

print("Uploading metadata to S3")
s3.upload_to_s3(bucket_name, bucket_dir_path, local_dir_path)
print("Uploading finished")
