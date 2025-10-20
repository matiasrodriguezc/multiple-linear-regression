# Multiple Linear Regression Examples in Python

This repository contains several examples of multiple linear regression implemented in Python, showcasing two powerful libraries: **Statsmodels** and **Scikit-learn**. The scripts cover models with both continuous and categorical (dummy) variables.

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

These scripts use `scikit-learn` to build predictive models, focusing on feature selection, model fitting, and making predictions.

  * **`gpa_prediction_sklearn.py`**: A model to predict GPA using SAT scores and a random variable to demonstrate feature selection.

      * **Dataset**: `1.02. Multiple linear regression.csv`
      * **Independent Variables (X)**: `SAT`, `Rand 1,2,3`
      * **Dependent Variable (Y)**: `GPA`

  * **`real_estate_prediction_sklearn.py`**: Predicts real estate prices based on property size and the year of construction.

      * **Dataset**: `real_estate_price_size_year.csv`
      * **Independent Variables (X)**: `size`, `year`
      * **Dependent Variable (Y)**: `price`

-----

## 🛠️ Key Concepts Demonstrated

  * **Data Preprocessing**: Converting categorical data (`Yes`/`No`) into numerical dummy variables (1/0).
  * **Multiple Regression**: Fitting models with multiple independent variables using both `statsmodels.api.OLS` and `sklearn.linear_model.LinearRegression`.
  * **Model Interpretation**: Analyzing the detailed statistical output from `statsmodels`.
  * **Feature Selection**: Using `sklearn.feature_selection.f_regression` to calculate F-statistics and p-values to evaluate feature importance.
  * **Model Evaluation**: Calculating R-squared and creating a function for Adjusted R-squared to assess model performance.
  * **Prediction**: Using a trained `scikit-learn` model to make predictions on new data.

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

**For the Scikit-learn examples:**

```bash
# GPA prediction model
python3 gpa_prediction_sklearn.py

# Real estate price prediction model
python3 real_estate_prediction_sklearn.py
```

-----

## 📈 Analysis of Results

Each script provides a different kind of output:

  * The **`statsmodels` script** prints a detailed summary table, ideal for understanding the significance and confidence intervals of each coefficient. It also visualizes how a dummy variable creates separate regression lines for each category.
  * The **`scikit-learn` scripts** focus on practical application. They print the model's coefficients and demonstrate how to build a summary table with p-values to help decide which features to keep in a predictive model. They also show how to use the final model to make predictions.