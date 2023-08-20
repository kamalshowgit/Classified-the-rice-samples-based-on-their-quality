# Rice Variety Classification using Naive Bayes Classifier

This project involves the implementation of a Naive Bayes classifier for classifying rice varieties as "Cammeo" or "Osmancik". The classifier is trained and tested on a given dataset, and its performance is evaluated using various metrics.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Project Overview](#project-overview)
- [Functions](#functions)
  - [`cross_validation_split`](#cross_validation_split)
  - [`gaussian_function`](#gaussian_function)
  - [`train`](#train)
  - [`predict_and_test`](#predict_and_test)
  - [`print_res`](#print_res)
  - [`predict_and_test_sklearn`](#predict_and_test_sklearn)
- [Usage](#usage)
- [Results](#results)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Redundant Feature Analysis](#redundant-feature-analysis)
- [Conclusion](#conclusion)

## Introduction

This project focuses on using a Naive Bayes classifier to differentiate between two rice varieties, namely "Cammeo" and "Osmancik". The goal is to build a classification model that can accurately classify new instances of rice based on their features.

## Dependencies

This project requires the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`

## Project Overview

1. Data Reading: The dataset is read from the CSV file named "Rice_Cammeo_Osmancik.csv".

2. Data Preprocessing: The dataset is shuffled and split into training and testing sets (70% training, 30% testing).

3. Cross-Validation: The training data is further split into 5 folds for cross-validation.

4. Model Training: The Naive Bayes classifier is trained on each fold of the training data using the provided functions.

5. Model Testing and Evaluation: The trained models are tested on the remaining 30% of the data, and evaluation metrics like accuracy, precision, recall, and F1-score are calculated.

6. Sklearn Comparison: A Naive Bayes model is also trained using the `sklearn` library for comparison and evaluation.

7. Hyperparameter Tuning: The impact of removing certain features on model performance is analyzed.

8. Redundant Feature Analysis: The impact of removing individual features on model performance is analyzed.

## Functions

### `cross_validation_split`

This function splits the dataset into `n_folds` parts for cross-validation.

### `gaussian_function`

This function calculates the Gaussian probability density function.

### `train`

This function trains the Naive Bayes classifier using the training data and calculates necessary probabilities and parameters.

### `predict_and_test`

This function predicts the class labels for the testing data using the trained Naive Bayes classifier and evaluates the model's performance.

### `print_res`

This function prints the evaluation results in a tabular form.

### `predict_and_test_sklearn`

This function trains and tests a Naive Bayes classifier using the `sklearn` library for comparison.

## Usage

1. Ensure you have the required libraries installed.
2. Place the dataset CSV file named "Rice_Cammeo_Osmancik.csv" in the same directory as the script.
3. Run the script to perform the classification and evaluation.

## Results

The project outputs evaluation results for the Naive Bayes classifier trained from scratch and the one trained using `sklearn`. Metrics like accuracy, precision, recall, and F1-score are provided.

## Hyperparameter Tuning

The impact of altering the features used for classification is analyzed, and the model's performance is evaluated.

## Redundant Feature Analysis

The impact of removing individual features on the model's performance is analyzed to identify any redundant features.

## Conclusion

This project demonstrates the implementation of a Naive Bayes classifier for rice variety classification. By comparing models trained from scratch and using `sklearn`, and by analyzing the impact of feature alterations, the project provides insights into the classification process and the significance of various features.