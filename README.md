# Diabetes-ML-Models

## Diabetes Detection Using Machine Learning

## Project Overview
This project aims to predict the likelihood of diabetes in patients based on various health metrics. Using a variety of machine learning models, this project explores the performance of different algorithms to detect diabetes with high accuracy.

## Aim
The primary goal of this project is to develop and compare various machine learning models for detecting diabetes based on clinical data. The models are trained to identify patterns and risk factors that can help in early detection and intervention for diabetes.

## Tech Stack
The project is developed using the following technologies:

- **Python 3.x**: Main programming language used for the implementation of machine learning models.
- **Jupyter Notebook**: For running and testing the code interactively.
- **NumPy**: For numerical computations and data manipulation.
- **Pandas**: For data processing and analysis.
- **Scikit-learn**: Machine learning library for implementing and evaluating the models.
- **Matplotlib & Seaborn**: For visualizing data trends and model results.
- **Joblib**: For model serialization and deserialization.

## Dataset
The dataset used for this project contains medical diagnostic measurements for various patients. It includes the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: A function which scores likelihood of diabetes based on family history
- **Age**: Age (years)
- **Outcome**: Class variable (0: No Diabetes, 1: Diabetes)

## Machine Learning Models Implemented

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Decision Tree**
4. **Support Vector Machine (SVM)**
5. **K-Nearest Neighbors (KNN)**
6. **Naive Bayes**
7. **Gradient Boosting**

## Results

- The models were evaluated using accuracy, precision, recall, and F1-score.
- The best-performing model was the **Random Forest Classifier** with an accuracy of approximately **X%**.
- The confusion matrix and ROC curve were plotted to visualize the performance of the models.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Diabetes-Detection.git
   ```
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Open the Jupyter notebook:
   ```bash
    pip install -r requirements.txt
   ```
4. Run all cells to train the models and view the results.

## Conclusions
This project demonstrates the application of machine learning models for predicting diabetes based on medical data. Future work may involve improving model accuracy, testing with larger datasets, and implementing additional models.


