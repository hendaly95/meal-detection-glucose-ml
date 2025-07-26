# 🧠 Meal Detection using Glucose Sensor Data

This project builds a machine learning pipeline to classify whether a person is having a meal based on glucose and insulin sensor data. It focuses on signal processing and time-series feature extraction from Continuous Glucose Monitoring (CGM) inputs to develop a robust binary classifier.

## 📌 Project Overview

- Applies FFT and statistical features on CGM signals.
- Uses a Decision Tree classifier for binary classification (Meal / No Meal).
- Outputs predictions to `Result.csv` for evaluation.

## 📁 Dataset

The original CGM and insulin dataset used in this project contains sensitive information and is therefore **not included** in this repository.  
To test or replicate the results, use a dataset with the following structure:

- `CGMSeriesLunch.csv`  
- `CGMSeriesNoLunch.csv`  
- `InsulinData.csv`

All files must have consistent time-stamped entries for proper alignment.

## 🚀 Features

- Feature extraction using:
  - FFT coefficients
  - First and second-order differences
  - Min, max, std over time windows
- Decision Tree classifier using scikit-learn
- 5-fold cross-validation
- Result export in CSV format

## 🗂️ File Descriptions

| File                   | Description                                                 |
|------------------------|-------------------------------------------------------------|
| `train.py`             | Extracts features, trains the model, saves trained classifier using `joblib` |
| `test.py`              | Loads trained model and makes predictions on test data      |
| `Result.csv`           | Final output containing binary predictions (1 = meal, 0 = no meal) |
| `Project Description.pdf` | Summary of the methodology and project stages            |

## 🛠️ How to Run

### Step 1: Install Dependencies

```bash
pip install numpy pandas scipy scikit-learn

### Step 2: Train the Model
Run the training script to preprocess data and train the classifier:

python train.py

This extracts features from the input CGM and insulin datasets, then trains a Decision Tree classifier.
The model is saved as a serialized .pkl file using joblib.

### Step 3: Test the Model
Use the testing script to generate predictions on unseen test data:

python test.py
This saves a file named Result.csv containing binary predictions.

📊 Sample Output: Result.csv

1
0
0
1
1
...
Each line represents the prediction for a test instance:

1 = meal detected

0 = no meal
