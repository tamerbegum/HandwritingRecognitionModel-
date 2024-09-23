# Handwriting Recognition Model

## Overview
Handwriting recognition model built with TensorFlow and Flask, utilizing a dataset of over 400,000 handwritten names. The project includes data preprocessing, model training, and a Flask API for real-time predictions.

## Dataset
The dataset is sourced from Kaggle and contains handwritten images with associated labels. 

The dataset consists of more than four hundred thousand handwritten names collected through charity projects. It includes a total of 206,799 first names and 207,024 surnames, divided into the following subsets:

- **Training Set**: 331,059 samples
- **Validation Set**: 41,382 samples
- **Testing Set**: 41,382 samples

The input data consists of hundreds of thousands of images of handwritten names. The dataset is organized into folders for train, validation, and test images, and includes the following CSV files:

- **written_name_train_v2.csv**: Contains the training data with filenames and identities.
- **written_name_validation_v2.csv**: Contains the validation data with filenames and identities.
- **written_name_test_v2.csv**: Contains the testing data with filenames and identities.

## Features
- **FILENAME**: The name of the image file such as TEST_0086.jpg
- **IDENTITY**: The corresponding text label for the handwritten image such as LEILI.

## Project Workflow

1. **Data Preprocessing**
   - Load the training and validation datasets.
   - Handle missing values and remove unreadable images.
   - Preprocess images by resizing and normalizing them for model input.

2. **Model Training**
   - Define a Convolutional Neural Network (CNN) architecture followed by Bidirectional LSTM layers.
   - Use CTC (Connectionist Temporal Classification) for training the model on image sequences.
   - Train the model and save the best weights for later use.

3. **Prediction**
   - Load the trained model for inference.
   - Implement a Flask API to accept image uploads and return predictions.


## Results

The current results indicate potential, and improvements are expected.

## How to Run

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
3. Start the Flask server:
   ```bash
   python app.py
4. Access the application: Open your browser and navigate to http://localhost:5999
