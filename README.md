# Soil Texture Prediction

Source code to build and train models to make predictions of soil texture and organic matter content using soil images. Quick description of the project:

- `src/cloudScripts/s3.py:` File with AWS S3 client connection logic. S3 is used to save the models when trained using AWS EC2 instances.
- `src/cloudScripts/2B.py:` Script to train 2B models in AWS EC2 instance.
- `src/cloudScripts/camera.py:` Script to train camera models in AWS EC2 instance.
- `src/cloudScripts/microscope.py:` Script to train microscope models in AWS EC2 instance.
- `src/models/customDataset.py:` File to build datasets from images. The structure of the csv file with lables is described in comments in the code.
- `src/models/models.py:` File with model classes and helper functions.
- `src/models/2B.ipynp:` Jupyter notebook to test 2B model.
- `src/models/camera.ipynp:` Jupyter notebook to test camera model.
- `src/models/microscope.ipynp:` Jupyter notebook to test microscope model.
- `src/models/labelsInfo.csv:` CSV file with test samples information.
- `src/protoype/controlRoutine.py:` Prototype control routine used for obtaining images.

## Testing models

When testing the models, several considerations must be taken into account:

1. Image Naming Convention:
   - A maximum of 999 images (with a .jpg extension) can be tested at a time.
   - Each image file must follow a naming format of three-digit numbers with leading zeros (e.g., 003.jpg).
   - If a different naming convention or file extension is required, you can modify the `load_data` method in the `OneImageCropboxRotationDataset` and `TwoImagesCropboxRotationDataset` classes.
2. Directory Organization:
   - Separate directories are required for storing camera images and microscope images.
3. CSV File with Labels:
   - A CSV file containing labels for each sample must be provided to construct the datasets.
   - This file is not used during the prediction process but is required afterward to calculate errors by comparing predictions with true values.
   - If true values are not available, you can set all values in the CSV file to 0.0.
   - Each row in the CSV must correspond to a pair of camera and microscope images.
   - For example, the file `src/models/labelsInfo.csv` demonstrates the structure of this file for two samples (001.jpg and 002.jpg).

## Notes

To use this project for training new models an adecuate environment must be setup in AWS EC2 instance. It can be used in other environments but some changes, regarding file locations, permissions, data location etc., will be needed.
