# Load in the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso

# Import custom preprocessing function
from preprocess import process_data

def preprocess_and_train_svr_model():
    data = process_data('./ml_data.csv', './uscities.csv')

    X = data[['Distance (miles)', 'Statetostate']]
    y = data['TotalRate']

    # We need to scale these features since SVM is sensitive to the scale of input data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 6)

    param_grid = {
        'C': [0.1, 1, 10, 100, 1000, 1500],
        'epsilon': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000, 1500],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }

    grid_search = GridSearchCV(SVR(), param_grid, cv=10, scoring='r2', verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best SVR model
    best_svr = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_svr.predict(X_test)

    # Eval the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Best Parameters:", grid_search.best_params_)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R²):", r2)

    # Plot actual vs predicted rates
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual TotalRate')
    plt.ylabel('Predicted TotalRate')
    plt.title('Actual vs Predicted TotalRate (SVR)')
    plt.grid(True)
    plt.show()

preprocess_and_train_svr_model()