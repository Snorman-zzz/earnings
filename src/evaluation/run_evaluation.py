import os
import sys
import time
from dotenv import load_dotenv
from llm_earnings_evaluator import LLMEarningsEvaluator

# Load environment variables
load_dotenv()


def main():
    # Get the API key - first try environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # If not in environment, prompt the user
    if not api_key:
        print("OPENAI_API_KEY not found in environment.")
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            sys.exit(1)
        # Set it for the current session
        os.environ["OPENAI_API_KEY"] = api_key

    # Display start banner
    print("\n" + "=" * 80)
    print("                  LLM STOCK PRICE PREDICTION EVALUATOR")
    print("                      FULL DATASET EVALUATION MODE")
    print("=" * 80 + "\n")

    # CSV files to evaluate
    csv_files = [
        "by_random_100.csv",
        "by_random_1000.csv",
        "by_symbol_random_10.csv",
        "by_symbol_time_10.csv",
        "by_time_100.csv"
    ]

    print(f"Files to process: {len(csv_files)}")
    for idx, file in enumerate(csv_files):
        print(f"  {idx + 1}. {file}")
    print()

    # Create evaluator
    evaluator = LLMEarningsEvaluator(
        test_data_dir='data',
        output_dir='result',
        use_cache=True  # Enable caching to avoid repeated API calls
    )

    # Start time for overall process
    start_time = time.time()

    # Process each file
    results = {}
    total_files = len(csv_files)

    for idx, csv_file in enumerate(csv_files):
        file_path = os.path.join('data', csv_file)

        # Progress header
        progress = f"[{idx + 1}/{total_files}]"
        print(f"\n{'=' * 80}")
        print(f"{progress} EVALUATING: {csv_file}")
        print(f"{'=' * 80}")

        if not os.path.exists(file_path):
            print(f"WARNING: File not found: {file_path}")
            continue

        file_start_time = time.time()

        try:
            # Run the evaluation with no row limit (0 means all rows)
            file_results = evaluator.run_evaluation(file_path, limit_rows=0)
            if file_results:
                results[csv_file] = file_results

                # Calculate file processing time
                file_time = time.time() - file_start_time

                # Print a summary of the results
                print(f"\n{progress} SUMMARY for {csv_file}:")
                print(f"Processing time: {file_time:.2f} seconds")

                # Access the accuracy from the direction metrics
                dir_metrics = file_results.get('direction_metrics', [])
                for metric in dir_metrics:
                    if metric.get('Symbol') == 'Total':
                        accuracy = metric.get('Direction Accuracy (%)', 0)
                        total_count = metric.get('Count', 0)
                        print(f"Rows processed: {total_count}")
                        print(f"Overall Direction Accuracy: {accuracy:.2f}%")
                        break

                # Access profit metrics
                profit = file_results.get('profit_metrics', {})
                if profit:
                    winning_rate = profit.get('winning_rate', 0)
                    actual_profit = profit.get('actual_profit', 0)
                    winning_trades = profit.get('winning_trades', 0)
                    losing_trades = profit.get('losing_trades', 0)
                    print(f"Winning Rate: {winning_rate:.2f}% ({winning_trades} trades)")
                    print(f"Losing Rate: {100 - winning_rate:.2f}% ({losing_trades} trades)")
                    print(f"Actual Profit: ${actual_profit:.2f}")

                print(f"{'=' * 80}")
        except Exception as e:
            print(f"ERROR processing {csv_file}: {str(e)}")

    # Calculate total elapsed time
    total_time = time.time() - start_time

    # Print overall summary
    print("\n\n" + "=" * 80)
    print("                  COMPLETE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total processing time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Files processed: {len(results)}/{total_files}")
    print(f"API cache entries: {len(evaluator.prediction_cache)}")
    print("\nPERFORMANCE BY FILE:")
    print("=" * 80)

    # Create a summary table with fixed width columns
    print(f"{'FILE':<20} {'DIRECTION ACC':<15} {'WINNING RATE':<15} {'PROFIT':<15} {'ROWS':<10}")
    print("-" * 80)

    for file_name, file_results in results.items():
        # Direction accuracy & row count
        dir_accuracy = "N/A"
        row_count = "N/A"
        dir_metrics = file_results.get('direction_metrics', [])
        for metric in dir_metrics:
            if metric.get('Symbol') == 'Total':
                dir_accuracy = f"{metric.get('Direction Accuracy (%)', 0):.2f}%"
                row_count = f"{metric.get('Count', 0)}"
                break

        # Profit metrics
        profit = file_results.get('profit_metrics', {})
        winning_rate = f"{profit.get('winning_rate', 0):.2f}%" if profit else "N/A"
        actual_profit = f"${profit.get('actual_profit', 0):.2f}" if profit else "N/A"

        # Print the row in table format
        print(f"{file_name:<20} {dir_accuracy:<15} {winning_rate:<15} {actual_profit:<15} {row_count:<10}")

    print("\nEvaluation completed for all available files.")


if __name__ == "__main__":
    main()