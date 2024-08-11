# soiltextureprediction

Source code to build and train models to make predictions of soil texture and organic matter content using soil images. Quick description of the project:

- src/customDataset.py -> File to build datasets from images
- src/models.py -> File to build models
- src/s3 -> File with AWS S3 client connection logic
- src/cloudTrainingScript -> Files to train models in AWS EC2 instance.

To use this project an adecuate environment must be setup in AWS EC2 instance. It can be used in other environments but some changes, regarding file locations, permissions, data location etc., will be needed.
