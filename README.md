# Multiple Linear Regression with a Dummy Variable

This project demonstrates how to build and interpret a multiple linear regression model that includes a **dummy variable**. The script uses Python with the `statsmodels`, `pandas`, and `matplotlib` libraries to analyze the relationship between a student's GPA, their SAT score, and their class attendance.

-----

## 🎯 Project Goal

The primary goal is to determine if class attendance (a categorical variable) has a statistically significant effect on a student's GPA, even after accounting for their SAT score (a continuous variable). This is achieved by converting the 'Attendance' category into a numerical dummy variable.

-----

## 📊 Dataset

The script uses the `1.03. Dummies.csv` dataset, which contains the following columns:

  * **`SAT`**: The student's score on the SAT (Independent Variable, continuous).
  * **`GPA`**: The student's Grade Point Average (Dependent Variable, continuous).
  * **`Attendance`**: Whether the student attended more than 75% of classes (`Yes`/`No`) (Independent Variable, categorical).

-----

## 🛠️ Key Concepts Demonstrated

  * **Data Preprocessing**: Converting categorical data ('Yes'/'No') into a binary format (1/0) using the `.map()` function in pandas.
  * **Dummy Variables**: Understanding how to incorporate non-numerical data into a regression model.
  * **Multiple Linear Regression**: Fitting a model with more than one independent variable using `statsmodels.api.OLS`.
  * **Model Interpretation**: Analyzing the `statsmodels` summary to understand the coefficients, p-values, and overall model fit.
  * **Data Visualization**: Plotting the regression results to show how the dummy variable creates separate regression lines for each category.
  * **Prediction**: Using the fitted model to make predictions on new, unseen data.

-----

## ⚙️ Requirements

To run this script, you will need Python 3 and the following libraries:

  * `numpy`
  * `pandas`
  * `statsmodels`
  * `matplotlib`
  * `seaborn`

-----

## 🚀 Installation

1.  Make sure you have Python 3 installed.
2.  Install the required libraries by running the following command in your terminal:
    ```bash
    python3 -m pip install numpy pandas statsmodels matplotlib seaborn
    ```

-----

## ▶️ How to Run the Script

1.  Save the code as a Python file (e.g., `dummy_variable_regression.py`).
2.  Place the `1.03. Dummies.csv` file in the same directory.
3.  Execute the script from your terminal:
    ```bash
    python3 dummy_variable_regression.py
    ```

-----

## 📈 Analysis of Results

The script will first print a summary of the regression model to the console. The key takeaway from the model is the equation:

$$GPA = b_0 + b_1 \cdot SAT + b_2 \cdot Attendance$$

The script then generates visualizations that plot two parallel regression lines:

  * **Did Not Attend (`Attendance = 0`)**: $GPA = b_0 + b_1 \cdot SAT$
  * **Attended (`Attendance = 1`)**: $GPA = (b_0 + b_2) + b_1 \cdot SAT$

The vertical distance between these two lines visually represents the coefficient $b_2$, which is the "bonus" or "penalty" to GPA associated with attendance, holding the SAT score constant. The final plots color the data points by attendance to make this relationship clear.

Finally, the script demonstrates how to use the `results.predict()` method to predict the GPA for new students with given SAT scores and attendance records.
