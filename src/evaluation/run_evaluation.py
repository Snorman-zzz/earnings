import os
import sys
from dotenv import load_dotenv
from llm_earnings_evaluator import LLMEarningsEvaluator

# Load environment variables
load_dotenv()


def main():
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment variables.")
        sys.exit(1)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run LLM Earnings Evaluation')
    parser.add_argument('--file', type=str, default="by_random_100.csv",
                        help='CSV file to evaluate (default: by_random_100.csv)')
    parser.add_argument('--limit', type=int, default=10,
                        help='Limit number of rows to process (default: 10, use 0 for all rows)')
    args = parser.parse_args()

    # Create evaluator
    evaluator = LLMEarningsEvaluator(
        test_data_dir='data',
        output_dir='result',
        use_cache=True  # Enable caching to avoid repeated API calls
    )

    # Run evaluation on the specified file
    file_path = os.path.join('data', args.file)
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    print(f"Running evaluation on {args.file}")
    print(f"Processing {args.limit if args.limit > 0 else 'all'} rows")

    # Run the evaluation
    results = evaluator.run_evaluation(file_path, limit_rows=args.limit)

    if results:
        print("\nEvaluation completed successfully!")

        # Access the accuracy from the direction metrics
        dir_metrics = results.get('direction_metrics', [])
        for metric in dir_metrics:
            if metric.get('Symbol') == 'Total':
                accuracy = metric.get('Direction Accuracy (%)', 0)
                print(f"\nOverall Direction Accuracy: {accuracy:.2f}%")
                break

        # Access profit metrics
        profit = results.get('profit_metrics', {})
        if profit:
            winning_rate = profit.get('winning_rate', 0)
            actual_profit = profit.get('actual_profit', 0)
            print(f"Winning Rate: {winning_rate:.2f}%")
            print(f"Actual Profit: ${actual_profit:.2f}")
    else:
        print("Evaluation failed or returned no results")


if __name__ == "__main__":
    main()