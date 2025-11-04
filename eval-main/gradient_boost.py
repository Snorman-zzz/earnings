import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

def prepare_features(df):
    """
    Prepare features for the model from raw earnings data.
    
    Args:
        df (pd.DataFrame): DataFrame with earnings data
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    
    return df[['Close', 'EPS Estimate', 'Reported EPS', 'Surprise(%)']]

def train_model(df, random_state=42, grid_search=False):
    """
    Train a gradient boosting model on earnings data.
    
    Args:
        df (pd.DataFrame): DataFrame with earnings data
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (trained_model, feature_names)
    """
    # Prepare features
    X_train = prepare_features(df)
    y_train = df['Open Diff(%)']
    # First model: Direction classifier
    y_direction = (y_train > 0).astype(int)  # Convert to binary (up=1, down=0)

    direction_model = HistGradientBoostingClassifier()

    # grid search for hyperparameters
    if grid_search:
        param_grid = {
            'max_iter': [50, 100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_leaf': [10, 20, 30]
        }
        grid_search = GridSearchCV(direction_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_direction)
        direction_model = grid_search.best_estimator_
        print(f"Best parameters for direction model: {grid_search.best_params_}")
    else:
        direction_model.fit(X_train, y_direction)

    # evaluate the model
    y_pred = direction_model.predict(X_train)
    accuracy = np.mean(y_pred == y_direction)
    print(f"Direction model accuracy: {accuracy:.2f}")

    # Second model: Magnitude regressor
    if grid_search:
        param_grid = {
            'max_iter': [50, 100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_leaf': [10, 20, 30]
        }
        grid_search = GridSearchCV(HistGradientBoostingRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, np.abs(y_train))
        magnitude_model = grid_search.best_estimator_
        print(f"Best parameters for magnitude model: {grid_search.best_params_}")
    else:
        magnitude_model = HistGradientBoostingRegressor()
        magnitude_model.fit(X_train, np.abs(y_train))  # Predict absolute magnitude

    # evaluate the model
    y_pred = magnitude_model.predict(X_train)
    mae = np.mean(np.abs(y_pred - np.abs(y_train)))
    mse = mean_squared_error(np.abs(y_train), y_pred)
    r2 = r2_score(np.abs(y_train), y_pred)
    print(f"Magnitude model MAE: {mae: .2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

    return (direction_model, magnitude_model), X_train.columns.tolist()


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the gradient boosting model.
    
    Args:
        model: Trained machine learning model
        feature_names (list): List of feature names
    """
    # Extract feature importance from the model
    gbr = model[1]  # Use the regressor for feature importance
    importance = gbr.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    # Plot top 20 features or all if less than 20
    n_features = min(20, len(feature_names))
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(n_features), importance[indices[:n_features]], align='center')
    plt.xticks(range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def save_model(model, filename='gradient_boost_model.pkl'):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained machine learning model
        filename (str): Filename to save the model
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(filename='gradient_boost_model.pkl'):
    """
    Load a trained model from a file.
    
    Args:
        filename (str): Filename of the saved model
        
    Returns:
        Trained machine learning model
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

def predict(model, df):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained machine learning model
        df (pd.DataFrame): DataFrame with earnings data
        
    Returns:
        pd.DataFrame: DataFrame with original data and predictions
    """
    # Prepare features
    X = prepare_features(df)
    
    # Make predictions
    directions = model[0].predict(X)
    magnitudes = model[1].predict(X)
    
    # Add predictions to original dataframe
    result_df = df.copy()
    result_df['Pred_Change_Pct'] = np.where(directions == 1, magnitudes, -magnitudes)  # Apply direction
    
    # Calculate predicted price change percentage
    result_df['Post Open_pred'] = result_df['Close'] * (1 + result_df['Pred_Change_Pct'] / 100)
    
    return result_df[['Symbol', 'date', 'Post Open_pred', 'Pred_Change_Pct']]

if __name__ == "__main__":

    # Load the dataset
    train_data = pd.read_csv("data/train.csv")
    
    # Train the model
    print("\nTraining gradient boosting model...")
    model, feature_names = train_model(train_data, grid_search=True)
    
    # Generate predictions for all data
    print("\nGenerating predictions...")
    os.makedirs("data/predictions", exist_ok=True)
    
    # Load the test dataset
    for test_file in os.listdir("data/test"):
        test_data = pd.read_csv(os.path.join("data/test", test_file))
        print(f"Loaded test data for {test_data['Symbol'].nunique()} symbols")
        
        # Predict using the trained model
        predictions_df = predict(model, test_data)
        
        # Save predictions
        predictions_df.to_csv(os.path.join("data/predictions", test_file), index=False)