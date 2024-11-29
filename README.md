# Mobile Price Prediction with Machine Learning
This project predicts the price range of mobile phones using various machine learning models. The dataset includes features like battery power, RAM, screen size, and other hardware specifications to train and evaluate models.

## Table of Contents
Introduction,
Dataset,
Features,
Technologies Used,
Models Implemented,
Performance Metrics,
Usage,
Results,

## Introduction
Mobile phones have a wide range of features, and predicting their price range based on these features can help manufacturers, sellers, and customers. This project explores machine learning techniques to classify mobile phones into specific price ranges.

## Dataset
The dataset contains specifications of mobile phones and their respective price range categories.

Training data: Includes labels (price_range) for supervised learning.
Testing data: Excludes labels for validation.

## Key Features:
battery_power: Battery capacity in mAh.
ram: RAM size in MB.
fc and pc: Front and rear camera megapixels.
px_height and px_width: Screen resolution.
int_memory: Internal memory in GB.
n_cores: Number of processor cores.
four_g and dual_sim: Support for 4G and dual SIM.

## Technologies Used
Python: Primary programming language.
Scikit-learn: Machine learning library for model implementation and evaluation.
XGBoost: For boosted tree classification.
Matplotlib: Visualization of confusion matrices.
Google Colab: Execution environment.

## Models Implemented
The following machine learning models were tested:

Support Vector Classifier (SVC)
Logistic Regression
Random Forest Classifier
Decision Tree Classifier
K-Nearest Neighbors (KNN)
XGBoost Classifier
AdaBoost Classifier
Gradient Boosting Classifier
Gaussian Process Classifier

## Performance Metrics
Accuracy was used as the primary metric to compare models. Confusion matrices were also plotted to analyze model predictions.

Model	Accuracy (%)
Support Vector Machine	89.00
Logistic Regression	95.50
Random Forest	81.75
Decision Trees	84.00
K-Nearest Neighbors	--
XGBoost	--
AdaBoost	--
Gradient Boosting	--
Gaussian Process	--

## Hyperparameter Tuning
Grid Search was applied to Logistic Regression, achieving:

Best parameters: {'C': 100, 'penalty': 'l2'}
Best accuracy: 97.5%

## Usage
Clone the repository:
git clone <repository_url>
Install dependencies:
pip install -r requirements.txt
Execute the script on Google Colab or your local environment:
python mobile_price_prediction.py

## Results
Logistic Regression achieved the highest accuracy with optimized hyperparameters (C=100, penalty='l2'). Support Vector Classifier and Decision Trees also demonstrated competitive performance.










