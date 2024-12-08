# Load in the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# Import custom preprocessing function
from preprocess import process_data

def preprocess_and_train_linear_regression_model():
    data = process_data('./ml_data.csv', './uscities.csv')

    X = data[['Distance (miles)', 'Statetostate']]
    y = data['TotalRate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get the residuals
    y_train_pred = model.predict(X_train)
    residuals = y_train - y_train_pred

    threshold = 3 * np.std(residuals)
    non_outliers = np.abs(residuals) <= threshold
    X_train_filtered = X_train[non_outliers]
    y_train_filtered = y_train[non_outliers]

    # Retrain the model without outliers
    model.fit(X_train_filtered, y_train_filtered)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Eval model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Perf metrics
    print("Mean Squared Error (MSE):", mse)
    print("R-squared:", r2)

    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    # Plot actual vs predicted rates
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual TotalRate')
    plt.ylabel('Predicted TotalRate')
    plt.title('Actual vs Predicted TotalRate (After Outlier Removal)')
    plt.grid(True)
    plt.show()

preprocess_and_train_linear_regression_model()
