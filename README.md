# Multiple Linear Regression Examples in Python

This repository contains several examples of multiple linear regression implemented in Python, showcasing two powerful libraries: **Statsmodels** and **Scikit-learn**. The scripts cover models with continuous, categorical (dummy), and **scaled** variables.

-----

## 📂 Scripts Overview

This project demonstrates both a statistical and a machine learning approach to multiple regression.

### 1\. Statsmodels Implementation (Statistical Inference)

This script is perfect for detailed statistical interpretation, providing comprehensive summaries for model analysis.

  * **`dummy_variable_regression_statsmodels.py`**: Analyzes the effect of SAT scores and class attendance (a dummy variable) on a student's GPA.
      * **Dataset**: `1.03. Dummies.csv`
      * **Independent Variables (X)**: `SAT`, `Attendance`
      * **Dependent Variable (Y)**: `GPA`

### 2\. Scikit-learn Implementations (Machine Learning & Prediction)

These scripts use `scikit-learn` to build predictive models, focusing on feature scaling, model fitting, and prediction.

  * **`gpa_prediction_scaled_sklearn.py`**: A model to predict GPA using SAT scores. This script introduces **feature scaling** to improve model interpretation.

      * **Dataset**: `1.02. Multiple linear regression.csv`
      * **Independent Variables (X)**: `SAT`, `Rand 1,2,3`
      * **Dependent Variable (Y)**: `GPA`

  * **`real_estate_prediction_scaled_sklearn.py`**: Predicts real estate prices by implementing feature scaling for the independent variables.

      * **Dataset**: `real_estate_price_size_year.csv`
      * **Independent Variables (X)**: `size`, `year`
      * **Dependent Variable (Y)**: `price`

-----

## 🛠️ Key Concepts Demonstrated

  * **Data Preprocessing**: Converting categorical data into numerical dummy variables.
  * **Feature Scaling (Standardization)**: Using `StandardScaler` to normalize features, making their coefficients (weights) directly comparable and improving model stability.
  * **Multiple Regression**: Fitting models with multiple independent variables using both `statsmodels.api.OLS` and `sklearn.linear_model.LinearRegression`.
  * **Feature Selection**: Using `f_regression` to calculate p-values and evaluate feature importance.
  * **Model Interpretation**: Analyzing statistical output from `statsmodels` and comparing coefficient magnitudes (weights) in scaled `scikit-learn` models.
  * **Prediction**: Using a trained model to make predictions, including the crucial step of **scaling new data** with the same scaler.

-----

## ⚙️ Requirements

To run these scripts, you will need Python 3 and the following libraries:

  * `numpy`
  * `pandas`
  * `statsmodels`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`

-----

## 🚀 Installation

1.  Make sure you have Python 3 installed.
2.  Install all the required libraries by running this command in your terminal:
    ```bash
    python3 -m pip install numpy pandas statsmodels matplotlib seaborn scikit-learn
    ```

-----

## ▶️ How to Run the Scripts

Navigate to the project directory in your terminal and execute the desired script.

**For the Statsmodels example:**

```bash
python3 dummy_variable_regression_statsmodels.py
```

**For the Scikit-learn examples with Feature Scaling:**

```bash
# GPA prediction model
python3 gpa_prediction_scaled_sklearn.py

# Real estate price prediction model
python3 real_estate_prediction_scaled_sklearn.py
```

-----

## 📈 Analysis of Results

Each script provides a different kind of output:

  * The **`statsmodels` script** prints a detailed summary table, ideal for understanding the statistical significance of each coefficient.
  * The **`scikit-learn` scripts** introduce **feature scaling**, a crucial preprocessing step in machine learning. By standardizing the features, we can directly compare the magnitude of their coefficients (often called 'weights') to assess feature importance. These scripts also highlight a critical best practice: any new data used for prediction **must be transformed using the same scaler** that was fitted on the training data.