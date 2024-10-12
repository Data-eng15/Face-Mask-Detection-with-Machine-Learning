# Face Mask Detection Dataset

## Overview

This dataset contains images of individuals **with masks** and **without masks**, which can be used to train machine learning models for face mask detection. The dataset is suitable for tasks like image classification, object detection, and real-time monitoring applications. It has been provided by Wobot Intelligence to assist in developing solutions to monitor mask compliance in public spaces.

## Dataset Structure

The dataset is organized into two main folders:

- `with_mask`: Images of people wearing face masks.
- `without_mask`: Images of people not wearing face masks.

Each folder contains several JPEG images that can be used for training, validation, and testing purposes.

### Folder Structure:

- **Total images**: (Specify number of images if known)
- **Resolution**: Images are in varying resolutions but can be resized to a standard size (e.g., 128x128 or 224x224) for model training.
- **Format**: JPEG, PNG

## Sample Images

Here are a few sample images from the dataset:

- **With Mask**: ![with_mask_sample](path_to_sample_image)
- **Without Mask**: ![without_mask_sample](path_to_sample_image)

## Usage

The dataset can be used for:

- **Training face mask detection models**: Use a machine learning model, like a convolutional neural network (CNN), to classify the images into two categories: with mask and without mask.
- **Real-time face mask detection**: After training a model, you can use it for real-time detection through a webcam or video feed.

### Example Model Pipeline

1. **Pre-processing**: Resize images, normalize pixel values, and augment the data for improved training results.
2. **Model Training**: Use a CNN or transfer learning from pre-trained models like MobileNetV2 or ResNet50 to classify the images.
3. **Validation and Testing**: Evaluate model performance on the validation and test sets, checking accuracy, precision, recall, etc.

## Download

To download the dataset, visit the following Kaggle page:

[Face Mask Detection Dataset on Kaggle](https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset)

Alternatively, use the Kaggle API:
```bash
kaggle datasets download -d wobotintelligence/face-mask-detection-dataset
@misc{wobotintelligence_2020,
    title={Face Mask Detection Dataset},
    author={Wobot Intelligence},
    year={2020},
    howpublished={\url{https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset}}
}

This `README.md` for the dataset provides an explanation of the dataset structure, usage, and citation details, and is helpful for users looking to understand and download the dataset for their own projects. You can adapt it as needed.
