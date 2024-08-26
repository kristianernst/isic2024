# ISIC 2024 Skin Cancer Classification Challenge

## Overview

This project is a play project for the ISIC 2024 Skin Cancer Classification Challenge. The goal is to classify skin lesions as benign or malignant using various machine learning models, including CNNs and Vision Transformers.

I also played around with self-supervised learning and contrastive learning.

## Project Structure

- **src/**: Contains all the source code for the project.
  - **data/**: Scripts for data fetching and preprocessing.
  - **model/**: Implementation of different models used in the project.
  - **utils/**: Utility functions for data handling, model saving, and other helper functions.
  - **train.py**: Main script for training the models.
  - **eval.py**: Script for evaluating the trained models.
  - **train_encoder.py**: Script for training the encoder using self-supervised learning techniques.

## Setup

### Requirements

The project dependencies are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Fetching Data

Fetch the data using the `fetch_data.py` script allows you to reformat a pandas dataframe with the data. I collected data from 2016-2024.

### Training the classifier

To train the models, use the `train.py` script. This script supports various models and configurations, which can be set through command-line arguments or a configuration file.

### Training the encoder

To train the encoder, use the `train_encoder.py` script. This script supports various models and configurations, which can be set through command-line arguments or a configuration file.

### Evaluation

To evaluate the trained models, use the `eval.py` script. This script loads the trained model and evaluates it on the validation set.
