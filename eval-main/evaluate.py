import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_price_predictions(actual_df, predictions_df):
    """
    Evaluate predictions of Post Open price based on earnings data.
    
    Args:
        actual_df (pd.DataFrame): DataFrame with actual data
        predictions_df (pd.DataFrame): DataFrame with predicted Post Open prices
        
    Returns:
        tuple: (regression_metrics_df, direction_metrics_df)
    """
    # Merge predictions with actual data
    merged_df = pd.merge(
        actual_df,
        predictions_df,
        on=['Symbol', 'date'],
        how='inner',
        suffixes=('', '_pred')
    )
    
    if len(merged_df) == 0:
        print("Error: No matching data between predictions and actual values")
        return None, None
    
    # Calculate price direction (up/down) for actual and predicted
    merged_df['Actual_Direction'] = (merged_df['Post Open'] > merged_df['Close']).astype(int)
    merged_df['Pred_Direction'] = (merged_df['Post Open_pred'] > merged_df['Close']).astype(int)
    
    # Calculate price change percentage for visualization
    
    results_regression = []
    results_direction = []
    
    # Evaluate by symbol
    for symbol in merged_df['Symbol'].unique():
        symbol_df = merged_df[merged_df['Symbol'] == symbol]
        
        # Regression metrics
        actual_prices = symbol_df['Post Open']
        pred_prices = symbol_df['Post Open_pred']
        
        rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
        mae = mean_absolute_error(actual_prices, pred_prices)
        r2 = r2_score(actual_prices, pred_prices) if len(actual_prices) > 1 else np.nan
        
        # Calculate percentage error for price predictions
        mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
        
        reg_metrics = {
            'Symbol': symbol,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Count': len(symbol_df)
        }
        results_regression.append(reg_metrics)
        
        # Direction metrics (binary classification)
        actual_dir = symbol_df['Actual_Direction']
        pred_dir = symbol_df['Pred_Direction']
        
        direction_accuracy = accuracy_score(actual_dir, pred_dir) * 100
        
        # Get confusion matrix values
        tn, fp, fn, tp = confusion_matrix(actual_dir, pred_dir, labels=[0, 1]).ravel()
        
        dir_metrics = {
            'Symbol': symbol,
            'Direction Accuracy (%)': direction_accuracy,
            'Correct Predictions': tp + tn,
            'Incorrect Predictions': fp + fn,
            'Up Predictions': tp + fp,
            'Down Predictions': tn + fn,
            'Actual Ups': tp + fn,
            'Actual Downs': tn + fp,
            'Count': len(symbol_df)
        }
        results_direction.append(dir_metrics)
        
        
    # Create summary DataFrames
    reg_df = pd.DataFrame(results_regression)
    dir_df = pd.DataFrame(results_direction)
    
    # Calculate total metrics
    if len(reg_df) > 0:
        reg_df.loc['Total', 'RMSE'] = np.sqrt(mean_squared_error(merged_df['Post Open'], merged_df['Post Open_pred']))
        reg_df.loc['Total', 'MAE'] = mean_absolute_error(merged_df['Post Open'], merged_df['Post Open_pred'])
        reg_df.loc['Total', 'MAPE'] = np.mean(np.abs((merged_df['Post Open'] - merged_df['Post Open_pred']) / merged_df['Post Open'])) * 100
        reg_df.loc['Total', 'R2'] = r2_score(merged_df['Post Open'], merged_df['Post Open_pred'])
        reg_df.loc['Total', 'Count'] = len(merged_df)
        reg_df.loc['Total', 'Symbol'] = 'Total'
        
    if len(dir_df) > 0:
        dir_df.loc['Total', 'Direction Accuracy (%)'] = accuracy_score(merged_df['Actual_Direction'], merged_df['Pred_Direction']) * 100
        dir_df.loc['Total', 'Correct Predictions'] = (merged_df['Actual_Direction'] == merged_df['Pred_Direction']).sum()
        dir_df.loc['Total', 'Incorrect Predictions'] = (merged_df['Actual_Direction'] != merged_df['Pred_Direction']).sum()
        dir_df.loc['Total', 'Up Predictions'] = (merged_df['Pred_Direction'] == 1).sum()
        dir_df.loc['Total', 'Down Predictions'] = (merged_df['Pred_Direction'] == 0).sum()
        dir_df.loc['Total', 'Actual Ups'] = (merged_df['Actual_Direction'] == 1).sum()
        dir_df.loc['Total', 'Actual Downs'] = (merged_df['Actual_Direction'] == 0).sum()
        dir_df.loc['Total', 'Count'] = len(merged_df)
        dir_df.loc['Total', 'Symbol'] = 'Total'
    
    return reg_df, dir_df

def evaluate_profit(actual_df, predictions_df):
    # if we buy 100 shares at the close price and sell at the post open price when the direction is up
    # calculate the profit according to the predictions
    merged_df = pd.merge(
        actual_df,
        predictions_df,
        on=['Symbol', 'date'],
        how='inner',
        suffixes=('', '_pred')
    )
    merged_df['Actual_Direction'] = (merged_df['Post Open'] > merged_df['Close']).astype(int)
    merged_df['Pred_Direction'] = (merged_df['Post Open_pred'] > merged_df['Close']).astype(int)
    merged_df['Ideal_Profit'] = np.where(
        merged_df['Actual_Direction'] == 1,
        (merged_df['Post Open'] - merged_df['Close']) * 100,
        0
    )
    merged_df['Predicted_Profit'] = np.where(
        merged_df['Pred_Direction'] == 1,
        (merged_df['Post Open_pred'] - merged_df['Close']) * 100,
        0
    )
    merged_df['Actual_Profit'] = np.where(
        merged_df['Pred_Direction'] == 1,
        (merged_df['Post Open'] - merged_df['Close']) * 100,
        0
    )

    print("\nProfit Evaluation:")
    total_ideal_profit = merged_df['Ideal_Profit'].sum()
    total_predicted_profit = merged_df['Predicted_Profit'].sum()
    total_actual_profit = merged_df['Actual_Profit'].sum()
    print(f"Total Ideal Profit: ${total_ideal_profit:.2f}")
    print(f"Total Predicted Profit: ${total_predicted_profit:.2f}")
    print(f"Total Actual Profit: ${total_actual_profit:.2f}")

    winning_trades = len(merged_df[(merged_df['Pred_Direction'] == 1) & (merged_df['Actual_Direction'] == 1)])
    losing_trades = len(merged_df[(merged_df['Pred_Direction'] == 1) & (merged_df['Actual_Direction'] == 0)])
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Winning Rate: {winning_trades / (winning_trades + losing_trades) * 100:.2f}%")



def create_prediction_plots(symbol_df, symbol):
    """
    Create visualizations for price and direction predictions.
    
    Args:
        symbol_df (pd.DataFrame): DataFrame with predictions for a single symbol
        symbol (str): Stock symbol for title
    """
    # Sort by date for time-series plots
    symbol_df = symbol_df.sort_values('date')
    
    # Plot 1: Actual vs Predicted Post-Earnings Opening Prices
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(symbol_df['date'], symbol_df['Post Open'], 'b-', label='Actual')
    plt.plot(symbol_df['date'], symbol_df['Post Open_pred'], 'r--', label='Predicted')
    plt.title(f'Post-Earnings Opening Price - {symbol}')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Percentage Change After Earnings
    plt.subplot(2, 1, 2)
    plt.bar(
        np.arange(len(symbol_df)) - 0.2, 
        symbol_df['Actual_Change_Pct'], 
        width=0.4, 
        label='Actual Change %',
        color='blue'
    )
    plt.bar(
        np.arange(len(symbol_df)) + 0.2, 
        symbol_df['Pred_Change_Pct'], 
        width=0.4, 
        label='Predicted Change %',
        color='red'
    )
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(np.arange(len(symbol_df)), symbol_df['date'], rotation=45)
    plt.title('Post-Earnings Price Change %')
    plt.ylabel('Price Change %')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'price_prediction_{symbol}.png')
    plt.close()
    
    # Create a scatter plot of actual vs predicted prices
    plt.figure(figsize=(8, 8))
    plt.scatter(symbol_df['Post Open'], symbol_df['Post Open_pred'], alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(symbol_df['Post Open'].min(), symbol_df['Post Open_pred'].min())
    max_val = max(symbol_df['Post Open'].max(), symbol_df['Post Open_pred'].max())
    padding = (max_val - min_val) * 0.05
    plt.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 'r--')
    
    plt.title(f'Actual vs. Predicted Post-Earnings Opening Price - {symbol}')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'price_correlation_{symbol}.png')
    plt.close()
    
    # Create a confusion matrix for direction predictions
    cm = confusion_matrix(symbol_df['Actual_Direction'], symbol_df['Pred_Direction'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Down', 'Up'],
        yticklabels=['Down', 'Up']
    )
    plt.title(f'Direction Prediction Confusion Matrix - {symbol}')
    plt.ylabel('Actual Direction')
    plt.xlabel('Predicted Direction')
    plt.tight_layout()
    plt.savefig(f'direction_cm_{symbol}.png')
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs("data/evaluation", exist_ok=True)
    
    # evaluate every test dataset under data/test
    # and save the results to csv files

    test_files = os.listdir("data/test")

    for test_file in test_files:    
        print(f"\nEvaluating predictions against {test_file}...")
        try:
            actual_df = pd.read_csv(os.path.join("data/test", test_file))
            predictions_df = pd.read_csv(os.path.join("data/predictions", test_file))
            
            print(f"Loaded actual data for {actual_df['Symbol'].nunique()} symbols")
            print(f"Loaded predictions for {predictions_df['Symbol'].nunique()} symbols")
            
            reg_metrics, dir_metrics = evaluate_price_predictions(actual_df, predictions_df)
            
            if reg_metrics is not None and dir_metrics is not None:
                print("\nRegression Metrics (Post Open Price Prediction):")
                print(reg_metrics.to_string())
                
                print("\nDirection Metrics (Up/Down Classification):")
                print(dir_metrics.to_string())
                
                # Save results
                reg_metrics.to_csv(f"data/evaluation/regression_metrics_{test_file}", index=False)
                dir_metrics.to_csv(f"data/evaluation/direction_metrics_{test_file}", index=False)

                # Evaluate profit
                evaluate_profit(actual_df, predictions_df)
                
        except Exception as e:
            import traceback
            print(f"Error during evaluation: {e}")
            print(traceback.format_exc())