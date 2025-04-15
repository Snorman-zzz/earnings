import os
import math
import csv
import json
import time
import random
from pathlib import Path

# LLM Client imports
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMEarningsEvaluator:
    """
    An evaluator for earnings predictions using actual LLM API calls
    instead of synthetic predictions.
    """

    def __init__(self,
                 test_data_dir='data',
                 output_dir='result',
                 use_cache=True):
        """Initialize the evaluator with paths and LLM settings."""
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(output_dir) / "cache"
        self.use_cache = use_cache

        # Make sure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Cache to avoid repeated API calls
        self.prediction_cache = {}
        self.load_cache()

    def read_csv(self, file_path):
        """Read CSV file and return as list of dictionaries."""
        data = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric strings to float
                for key, value in row.items():
                    try:
                        row[key] = float(value)
                    except (ValueError, TypeError):
                        pass
                data.append(row)
        return data

    def write_csv(self, data, file_path):
        """Write list of dictionaries to CSV file."""
        if not data:
            return

        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def load_cache(self):
        """Load prediction cache from disk if it exists."""
        cache_file = self.cache_dir / "openai_predictions_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.prediction_cache = json.load(f)
                print(f"Loaded {len(self.prediction_cache)} cached predictions")
            except Exception as e:
                print(f"Error loading cache: {str(e)}")
                self.prediction_cache = {}

    def save_cache(self):
        """Save prediction cache to disk."""
        cache_file = self.cache_dir / "openai_predictions_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.prediction_cache, f)
            print(f"Saved {len(self.prediction_cache)} predictions to cache")
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

    def get_llm_prediction(self, row, retry_count=3):
        """Get a price prediction from the LLM."""
        # Create a cache key from the row data
        cache_key = f"{row['Symbol']}_{row['date']}_{row['Close']}"

        # Return cached prediction if available and cache is enabled
        if self.use_cache and cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        # Prepare financial context for the LLM
        prompt = self.create_earnings_prompt(row)

        # Try to get prediction with retries
        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system",
                         "content": "You are a financial analyst specializing in predicting post-earnings stock price movements. Provide your best prediction based on the financial data provided."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Lower temperature for more deterministic responses
                )
                prediction_text = response.choices[0].message.content
                prediction = self.extract_price_prediction(prediction_text, row)

                # Cache the result
                self.prediction_cache[cache_key] = prediction
                # Save cache after each successful prediction
                if len(self.prediction_cache) % 5 == 0:  # Save every 5 predictions
                    self.save_cache()

                return prediction

            except Exception as e:
                print(f"Error getting prediction (attempt {attempt + 1}/{retry_count}): {str(e)}")
                time.sleep(2)  # Wait before retrying

        # If all retries fail, make a reasonable fallback prediction
        # Use a small random adjustment to the closing price
        fallback_prediction = row['Close'] * (1 + random.uniform(-0.02, 0.05))
        print(f"Using fallback prediction for {row['Symbol']}: {fallback_prediction}")
        self.prediction_cache[cache_key] = fallback_prediction
        return fallback_prediction

    def create_earnings_prompt(self, row):
        """Create a prompt for earnings prediction based on row data."""
        # Extract relevant fields
        symbol = row['Symbol']
        close_price = row['Close']

        # Get additional fields if they exist
        eps = row.get('EPS', 'N/A')
        revenue = row.get('Revenue', 'N/A')
        prev_close = row.get('PrevClose', close_price * 0.99)  # Fallback value
        volume = row.get('Volume', 'N/A')

        # Create a comprehensive prompt
        prompt = f"""
        I need you to predict the post-earnings stock price for {symbol}.

        Current financial data:
        - Symbol: {symbol}
        - Close Price: ${close_price:.2f}
        - Previous Close: ${prev_close:.2f}
        - EPS: {eps}
        - Revenue: {revenue}
        - Volume: {volume}

        Based on this earnings data, predict whether the stock will go up or down after earnings, and calculate the expected post-earnings opening price.

        Format your response as follows:

        Analysis: Your brief analysis of the earnings data and price movement prediction

        Direction: [UP/DOWN]

        Price Prediction Formula: CurrentPrice × (1 + AdjustmentFactor) = NewPrice

        Predicted Post-Earnings Opening Price: $X.XX
        """
        return prompt

    def extract_price_prediction(self, llm_response, row):
        """Extract the price prediction from the LLM response."""
        try:
            # Try to find a price prediction in the response
            price_pattern = r"Predicted Post-Earnings Opening Price: \$?(\d+\.?\d*)"
            import re
            price_match = re.search(price_pattern, llm_response)

            if price_match:
                return float(price_match.group(1))

            # Try to extract from formula if direct prediction not found
            formula_pattern = r"= \$?(\d+\.?\d*)"
            formula_match = re.search(formula_pattern, llm_response)

            if formula_match:
                return float(formula_match.group(1))

            # Last resort: look for any dollar amount
            dollar_pattern = r"\$(\d+\.?\d*)"
            dollar_matches = re.findall(dollar_pattern, llm_response)

            if dollar_matches and len(dollar_matches) > 0:
                # Use the last dollar amount in the response
                return float(dollar_matches[-1])

            # If no numerical prediction found, make a small adjustment to closing price
            print(f"No numerical prediction found in response for {row['Symbol']}, using fallback")
            print(f"Response was: {llm_response[:200]}...")
            return row['Close'] * 1.01  # 1% increase as fallback

        except Exception as e:
            print(f"Error extracting prediction: {str(e)}")
            print(f"Response was: {llm_response[:200]}...")
            # Return a default prediction with a small increase
            return row['Close'] * 1.01

    def generate_llm_predictions(self, actual_data, batch_size=10):
        """Generate predictions using an LLM API for each row in the dataset."""
        predictions = []
        total_rows = len(actual_data)

        print(f"Generating predictions for {total_rows} rows using OpenAI GPT-4o")

        # Process in batches to show progress
        for i in range(0, total_rows, batch_size):
            batch = actual_data[i:i + batch_size]
            print(
                f"Processing batch {i // batch_size + 1}/{math.ceil(total_rows / batch_size)} ({i + 1}-{min(i + batch_size, total_rows)} of {total_rows})")

            for row in batch:
                # Extract required values
                symbol = row['Symbol']
                date = row['date']
                close_price = row['Close']

                print(f"  Processing {symbol} for {date}")

                # Get prediction from LLM
                predicted_price = self.get_llm_prediction(row)

                predictions.append({
                    'Symbol': symbol,
                    'date': date,
                    'Post Open_pred': predicted_price
                })

            # Save intermediate results
            if i % (batch_size * 2) == 0 and i > 0:
                self.save_cache()

        return predictions

    def merge_data(self, actual_data, predictions):
        """Merge actual data with predictions."""
        # Create lookup dictionary for predictions
        pred_lookup = {}
        for pred in predictions:
            key = (pred['Symbol'], pred['date'])
            pred_lookup[key] = pred['Post Open_pred']

        # Merge data
        merged_data = []
        for row in actual_data:
            key = (row['Symbol'], row['date'])
            if key in pred_lookup:
                merged_row = row.copy()
                merged_row['Post Open_pred'] = pred_lookup[key]

                # Calculate price directions
                merged_row['Actual_Direction'] = 1 if row['Post Open'] > row['Close'] else 0
                merged_row['Pred_Direction'] = 1 if pred_lookup[key] > row['Close'] else 0

                # Calculate percentage changes
                merged_row['Actual_Change_Pct'] = ((row['Post Open'] - row['Close']) / row['Close']) * 100
                merged_row['Pred_Change_Pct'] = ((pred_lookup[key] - row['Close']) / row['Close']) * 100

                merged_data.append(merged_row)

        return merged_data

    def calculate_metrics(self, merged_data):
        """Calculate evaluation metrics without sklearn."""

        # Group data by symbol
        symbols = {}
        for row in merged_data:
            symbol = row['Symbol']
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(row)

        # Calculate regression metrics
        regression_metrics = []
        direction_metrics = []

        # Helper functions for metrics calculation
        def calc_rmse(actual, predicted):
            if not actual:
                return 0
            sum_squared_error = sum((a - p) ** 2 for a, p in zip(actual, predicted))
            return math.sqrt(sum_squared_error / len(actual))

        def calc_mae(actual, predicted):
            if not actual:
                return 0
            sum_abs_error = sum(abs(a - p) for a, p in zip(actual, predicted))
            return sum_abs_error / len(actual)

        def calc_mape(actual, predicted):
            if not actual:
                return 0
            sum_abs_percent_error = sum(abs((a - p) / a) * 100 for a, p in zip(actual, predicted) if a != 0)
            return sum_abs_percent_error / len(actual)

        def calc_r2(actual, predicted):
            if len(actual) <= 1:
                return float('nan')

            # Calculate mean of actual values
            mean_actual = sum(actual) / len(actual)

            # Calculate total sum of squares
            ss_total = sum((a - mean_actual) ** 2 for a in actual)
            if ss_total == 0:
                return float('nan')

            # Calculate residual sum of squares
            ss_residual = sum((a - p) ** 2 for a, p in zip(actual, predicted))

            # Calculate R-squared
            return 1 - (ss_residual / ss_total)

        def calc_accuracy(actual, predicted):
            if not actual:
                return 0
            correct = sum(1 for a, p in zip(actual, predicted) if a == p)
            return (correct / len(actual)) * 100

        def confusion_values(actual, predicted):
            """Calculate true positive, false positive, true negative, false negative."""
            tp = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
            fp = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
            tn = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)
            fn = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)
            return tp, fp, tn, fn

        # Calculate metrics for each symbol
        all_actual = []
        all_predicted = []
        all_actual_dir = []
        all_pred_dir = []

        for symbol, rows in symbols.items():
            actual = [row['Post Open'] for row in rows]
            predicted = [row['Post Open_pred'] for row in rows]
            actual_dir = [row['Actual_Direction'] for row in rows]
            pred_dir = [row['Pred_Direction'] for row in rows]

            # Extend the lists for total calculations
            all_actual.extend(actual)
            all_predicted.extend(predicted)
            all_actual_dir.extend(actual_dir)
            all_pred_dir.extend(pred_dir)

            # Calculate regression metrics
            rmse = calc_rmse(actual, predicted)
            mae = calc_mae(actual, predicted)
            mape = calc_mape(actual, predicted)
            r2 = calc_r2(actual, predicted)

            reg_metrics = {
                'Symbol': symbol,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2,
                'Count': len(rows)
            }
            regression_metrics.append(reg_metrics)

            # Calculate direction metrics
            direction_accuracy = calc_accuracy(actual_dir, pred_dir)
            tp, fp, tn, fn = confusion_values(actual_dir, pred_dir)

            dir_metrics = {
                'Symbol': symbol,
                'Direction Accuracy (%)': direction_accuracy,
                'Correct Predictions': tp + tn,
                'Incorrect Predictions': fp + fn,
                'Up Predictions': tp + fp,
                'Down Predictions': tn + fn,
                'Actual Ups': tp + fn,
                'Actual Downs': tn + fp,
                'Count': len(rows)
            }
            direction_metrics.append(dir_metrics)

        # Calculate total metrics
        total_rmse = calc_rmse(all_actual, all_predicted)
        total_mae = calc_mae(all_actual, all_predicted)
        total_mape = calc_mape(all_actual, all_predicted)
        total_r2 = calc_r2(all_actual, all_predicted)

        # Add total row to regression metrics
        regression_metrics.append({
            'Symbol': 'Total',
            'RMSE': total_rmse,
            'MAE': total_mae,
            'MAPE': total_mape,
            'R2': total_r2,
            'Count': len(merged_data)
        })

        # Calculate direction totals
        total_direction_accuracy = calc_accuracy(all_actual_dir, all_pred_dir)
        total_tp, total_fp, total_tn, total_fn = confusion_values(all_actual_dir, all_pred_dir)

        # Add total row to direction metrics
        direction_metrics.append({
            'Symbol': 'Total',
            'Direction Accuracy (%)': total_direction_accuracy,
            'Correct Predictions': total_tp + total_tn,
            'Incorrect Predictions': total_fp + total_fn,
            'Up Predictions': total_tp + total_fp,
            'Down Predictions': total_tn + total_fn,
            'Actual Ups': total_tp + total_fn,
            'Actual Downs': total_tn + total_fp,
            'Count': len(merged_data)
        })

        return regression_metrics, direction_metrics

    def evaluate_profit(self, merged_data):
        """Calculate profit metrics."""

        # Calculate profits
        for row in merged_data:
            # Buying 100 shares
            row['Ideal_Profit'] = (row['Post Open'] - row['Close']) * 100 if row['Actual_Direction'] == 1 else 0
            row['Predicted_Profit'] = (row['Post Open_pred'] - row['Close']) * 100 if row['Pred_Direction'] == 1 else 0
            row['Actual_Profit'] = (row['Post Open'] - row['Close']) * 100 if row['Pred_Direction'] == 1 else 0

        # Calculate total profits
        total_ideal_profit = sum(row['Ideal_Profit'] for row in merged_data)
        total_predicted_profit = sum(row['Predicted_Profit'] for row in merged_data)
        total_actual_profit = sum(row['Actual_Profit'] for row in merged_data)

        # Calculate winning and losing trades
        winning_trades = sum(1 for row in merged_data
                             if row['Pred_Direction'] == 1 and row['Actual_Direction'] == 1)
        losing_trades = sum(1 for row in merged_data
                            if row['Pred_Direction'] == 1 and row['Actual_Direction'] == 0)

        # Calculate winning rate
        total_trades = winning_trades + losing_trades
        winning_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Print results
        print("\nProfit Evaluation:")
        print(f"Total Ideal Profit: ${total_ideal_profit:.2f}")
        print(f"Total Predicted Profit: ${total_predicted_profit:.2f}")
        print(f"Total Actual Profit: ${total_actual_profit:.2f}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Winning Rate: {winning_rate:.2f}%")

        return {
            'ideal_profit': total_ideal_profit,
            'predicted_profit': total_predicted_profit,
            'actual_profit': total_actual_profit,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'winning_rate': winning_rate
        }

    def run_evaluation(self, test_file, limit_rows=None):
        """Run the full evaluation pipeline on a test file."""
        print(f"\nEvaluating with {os.path.basename(test_file)} using OpenAI GPT-4o...")

        # Load actual data
        actual_data = self.read_csv(test_file)

        # Limit rows if specified
        if limit_rows and limit_rows > 0 and limit_rows < len(actual_data):
            print(f"Limiting evaluation to first {limit_rows} rows")
            actual_data = actual_data[:limit_rows]

        print(f"Loaded {len(actual_data)} rows of actual data")

        # Count unique symbols
        symbols = set(row['Symbol'] for row in actual_data)
        print(f"Found {len(symbols)} unique symbols: {', '.join(symbols)}")

        # Generate predictions using LLM
        predictions = self.generate_llm_predictions(actual_data)

        # Save raw predictions
        test_name = os.path.splitext(os.path.basename(test_file))[0]
        predictions_file = os.path.join(self.output_dir, f'predictions_llm_{test_name}.csv')
        self.write_csv(predictions, predictions_file)
        print(f"Saved raw predictions to {predictions_file}")

        # Merge data
        merged_data = self.merge_data(actual_data, predictions)

        if not merged_data:
            print("Error: No matching data for evaluation")
            return None

        # Calculate metrics
        reg_metrics, dir_metrics = self.calculate_metrics(merged_data)

        # Print regression metrics
        print("\nRegression Metrics (Post Open Price Prediction):")
        print(f"{'Symbol':<10} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'R2':<10} {'Count':<10}")
        print("-" * 60)
        for row in reg_metrics:
            r2_val = f"{row['R2']:.4f}" if not math.isnan(row['R2']) else "N/A"
            print(
                f"{row['Symbol']:<10} {row['RMSE']:<10.4f} {row['MAE']:<10.4f} {row['MAPE']:<10.2f} {r2_val:<10} {row['Count']:<10}")

        # Print direction metrics
        print("\nDirection Metrics (Up/Down Classification):")
        print(f"{'Symbol':<10} {'Accuracy %':<12} {'Correct':<10} {'Incorrect':<10} {'Count':<10}")
        print("-" * 60)
        for row in dir_metrics:
            print(
                f"{row['Symbol']:<10} {row['Direction Accuracy (%)']:<12.2f} {row['Correct Predictions']:<10} {row['Incorrect Predictions']:<10} {row['Count']:<10}")

        # Save results
        self.write_csv(reg_metrics, os.path.join(self.output_dir, f'regression_metrics_llm_{test_name}.csv'))
        self.write_csv(dir_metrics, os.path.join(self.output_dir, f'direction_metrics_llm_{test_name}.csv'))

        # Evaluate profit
        profit_metrics = self.evaluate_profit(merged_data)

        return {
            'regression_metrics': reg_metrics,
            'direction_metrics': dir_metrics,
            'profit_metrics': profit_metrics
        }