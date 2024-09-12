# Comprehensive Linear Regression Analysis Documentation

This document provides a detailed explanation of a Python script that performs linear regression analysis on a dataset of heights and weights. The script demonstrates a complete workflow from data loading to model evaluation and prediction.

## Table of Contents
1. [Library Imports and Data Loading](#1-library-imports-and-data-loading)
2. [Data Exploration](#2-data-exploration)
3. [Data Visualization](#3-data-visualization)
4. [Feature Selection](#4-feature-selection)
5. [Data Splitting](#5-data-splitting)
6. [Feature Standardization](#6-feature-standardization)
7. [Linear Regression Model](#7-linear-regression-model)
8. [Model Visualization](#8-model-visualization)
9. [Predictions](#9-predictions)
10. [Model Evaluation](#10-model-evaluation)
11. [Advanced Statistics](#11-advanced-statistics)
12. [New Data Prediction](#12-new-data-prediction)
13. [Residual Analysis](#13-residual-analysis)
14. [Results](#14-residual-analysis)

## 1. Library Imports and Data Loading

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline

df = pd.read_csv(r'N:\Personal_Projects\Machine-Learning\linear_R\height-weight.csv')
```

- **pandas** (as pd): Used for data manipulation and analysis.
- **matplotlib.pyplot** (as plt): Used for creating static, animated, and interactive visualizations.
- **numpy** (as np): Used for numerical computing.
- **seaborn** (as sns): Used for statistical data visualization.
- **%matplotlib inline**: Jupyter notebook magic command to display plots inline.

The data is loaded from a CSV file into a pandas DataFrame named `df`.

## 2. Data Exploration

```python
df.head()
df.tail()
df.describe()
df.dtypes
```

These commands are used to inspect the dataset:
- `df.head()`: Displays the first 5 rows of the DataFrame.
- `df.tail()`: Displays the last 5 rows of the DataFrame.
- `df.describe()`: Provides statistical summary of the numerical columns (count, mean, std, min, 25%, 50%, 75%, max).
- `df.dtypes`: Shows the data types of each column in the DataFrame.

## 3. Data Visualization

```python
plt.scatter(df['Weight'], df['Height'])
plt.xlabel("Weight")
plt.ylabel(ylabel="Height")

df.corr()
sns.pairplot(df)
```

- The scatter plot visualizes the relationship between Weight (x-axis) and Height (y-axis).
- `df.corr()` calculates the correlation matrix between variables using Pearson correlation.
- `sns.pairplot(df)` creates a grid of scatter plots for each pair of variables in the dataset.

## 4. Feature Selection

```python
X = df[['Weight']]  # Independent variable
y = df['Height']    # Dependent variable
```

- `X` (independent variable): Weight, stored as a DataFrame.
- `y` (dependent variable): Height, stored as a Series.

Note: `X` is converted to a 2D array by using double square brackets `[[]]`. This is crucial for many scikit-learn functions that expect 2D arrays for X.

## 5. Data Splitting

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

- The data is split into training (75%) and testing (25%) sets.
- `random_state=42` ensures reproducibility of the split.

## 6. Feature Standardization

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

- StandardScaler is used to standardize features by removing the mean and scaling to unit variance.
- `fit_transform()` is used on the training data to compute the mean and std to be used for later scaling.
- `transform()` is used on the test data to scale it using the mean and std computed from the training data.

## 7. Linear Regression Model

```python
from sklearn.linear_model import LinearRegression
regression = LinearRegression(n_jobs=-1)
regression.fit(X_train, y_train)

print("Coefficient or slope:", regression.coef_)
print("Intercept:", regression.intercept_)
```

- LinearRegression model is instantiated with `n_jobs=-1` to use all available processors.
- The model is fitted to the training data using `fit()`.
- The coefficient (slope) and intercept of the regression line are printed.

## 8. Model Visualization

```python
plt.scatter(X_train, y_train)
plt.plot(X_train, regression.predict(X_train))
```

- This creates a scatter plot of the training data points.
- It then overlays the best-fit line obtained from the regression model.

## 9. Predictions

```python
y_pred = regression.predict(X_test)
```

- The fitted model is used to make predictions on the test set.

## 10. Model Evaluation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
score = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", score)
```

Several metrics are calculated to evaluate the model's performance:
- Mean Squared Error (MSE): Average squared difference between predicted and actual values.
- Mean Absolute Error (MAE): Average absolute difference between predicted and actual values.
- Root Mean Squared Error (RMSE): Square root of MSE, in the same units as the dependent variable.
- R-squared Score: Proportion of variance in the dependent variable predictable from the independent variable.

## 11. Advanced Statistics

```python
adjusted_r2 = 1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("Adjusted R-squared:", adjusted_r2)

import statsmodels.api as sm
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
```

- Adjusted R-squared is calculated, which adjusts for the number of predictors in the model.
- An Ordinary Least Squares (OLS) model is fitted using statsmodels, which provides a detailed statistical summary including p-values, confidence intervals, and various statistical tests.

## 12. New Data Prediction

```python
new_weight = regression.predict(scaler.transform([[72]]))
print("Predicted height for weight 72:", new_weight[0])
```

This demonstrates how to use the fitted model to predict height for a new weight value:
1. The new weight (72) is transformed using the same scaler used for the training data.
2. The scaled value is then passed to the model for prediction.

## 13. Residual Analysis

```python
residuals = y_test - y_pred
sns.distplot(residuals, kde=True)
```

- Residuals (differences between actual and predicted values) are calculated.
- A distribution plot of the residuals is created to check for normality and homoscedasticity.

## 14. Results

This section summarizes the key findings and results of the linear regression analysis:

### Model Parameters
- Coefficient (slope): [Value to be filled after running the code]
- Intercept: [Value to be filled after running the code]

### Model Performance Metrics
- Mean Squared Error (MSE): [Value to be filled after running the code]
- Mean Absolute Error (MAE): [Value to be filled after running the code]
- Root Mean Squared Error (RMSE): [Value to be filled after running the code]
- R-squared Score: [Value to be filled after running the code]
- Adjusted R-squared: [Value to be filled after running the code]

### Key Findings
1. Relationship between Weight and Height: [Brief description of the relationship observed in the scatter plot]
2. Model Fit: [Comment on how well the model fits the data based on R-squared and adjusted R-squared]
3. Prediction Accuracy: [Comment on the model's prediction accuracy based on RMSE and MAE]
4. Residual Analysis: [Brief description of the residual distribution and what it indicates about the model's assumptions]

### Practical Application
- Sample Prediction: For a weight of 72 units, the predicted height is [Value to be filled after running the code] units.

### Limitations and Future Work
- [Brief discussion of any limitations in the current analysis]
- [Suggestions for potential improvements or extensions to the analysis]
