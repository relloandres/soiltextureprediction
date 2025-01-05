# Soil Texture Prediction

Source code to build and train models to make predictions of soil texture and organic matter content using soil images. Quick description of the project:

- `src/models/customDataset.py:` File to build datasets from images. The structure of the csv file with lables is described in comments in the code.
- `src/models.py:` File to build models.
- `src/models/s3:` File with AWS S3 client connection logic. S3 is used to save the models when trained using AWS EC2 instances.
- `src/models/cloudScript:` Files to train models in AWS EC2 instance.
- `src/protoype/controlRoutine.py:` Prototype control routine used for obtaining images.
- `src/test:` Jupyter notebook with example code to test a 2B model.

To use this project for training new models an adecuate environment must be setup in AWS EC2 instance. It can be used in other environments but some changes, regarding file locations, permissions, data location etc., will be needed.
