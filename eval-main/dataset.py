from datetime import timedelta
import os
import pandas as pd
import numpy as np
import yfinance

pd.options.mode.chained_assignment = None


def get_stock_data(symbols, start_date, end_date, verbose=True):
    """
    Get stock data including earnings reports and post-earnings opening prices.
    
    Args:
        symbols (list): List of stock ticker symbols
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        verbose (bool): Whether to print progress information
        
    Returns:
        pd.DataFrame: Combined DataFrame with earnings and price data
    """
    all_df = []
    for i, symbol in enumerate(symbols):
        if verbose:
            print(f"Processing {symbol} ({i+1}/{len(symbols)})...")
        
        try:
            ticker = yfinance.Ticker(symbol)
            
            # Get more earnings dates by increasing the limit
            earnings_dates = ticker.get_earnings_dates(limit=100).dropna()
            if len(earnings_dates) == 0:
                if verbose:
                    print(f"No earnings data for {symbol}, skipping")
                continue
                
            earnings_dates['date'] = [_.date() for _ in earnings_dates.index]
            
            # Get price history
            history_price = ticker.history(start=start_date, end=end_date)
            if len(history_price) == 0:
                if verbose:
                    print(f"No price history for {symbol}, skipping")
                continue
                
            history_price = history_price.set_index(_.date() for _ in history_price.index)
            history_price['date'] = history_price.index

            # Merge price data with earnings data
            df = history_price[['date', 'Close']].merge(earnings_dates)
            
            # Skip if no earnings dates in our price history range
            if len(df) == 0:
                if verbose:
                    print(f"No earnings dates in price history range for {symbol}")
                continue
                
            # Calculate post-earnings opening price
            diff = timedelta(days=1)
            post_earning_days = [_ + diff for _ in df['date'].to_list()]
            
            # Filter to only include entries where we have post-earnings data
            valid_indices = []
            valid_post_days = []
            
            for idx, day in enumerate(post_earning_days):
                if day in history_price.index:
                    valid_indices.append(idx)
                    valid_post_days.append(day)
            
            if not valid_indices:
                if verbose:
                    print(f"No valid post-earnings days for {symbol}, skipping")
                continue
                
            # Keep only valid entries
            df = df.iloc[valid_indices].copy()
            
            # Add post-earnings opening price
            df['Post Open'] = history_price.loc[valid_post_days, 'Open'].to_list()
            
            # Calculate percentage change after earnings
            df['Open Diff(%)'] = (df['Post Open'] - df['Close']) / df['Close'] * 100
            
            # Add symbol column
            df['Symbol'] = symbol
            all_df.append(df)
            
            if verbose and len(df) > 0:
                print(f"  Added {len(df)} earnings events for {symbol}")
                
        except Exception as e:
            if verbose:
                print(f"Error processing {symbol}: {e}")
    
    if not all_df:
        if verbose:
            print("No data collected for any symbols")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_df, ignore_index=True)
    
    if verbose:
        print(f"Combined dataset has {len(combined_df)} records from {combined_df['Symbol'].nunique()} symbols")
    
    return combined_df

def split_test_dataset_by_time(cnt):
    df = pd.read_csv("data/combined_data.csv")
    if df.empty:
        print("Empty dataframe cannot be split")
        return
    
    # Sort by date for chronological split
    df = df.sort_values('date')
    
    # Split based on time
    test_df = df.iloc[len(df) - cnt:].copy()
    
    print(f"Choose {cnt} time based test data from {test_df['date'].min()} to {test_df['date'].max()}")

    test_df.to_csv(f"data/test/by_time_{cnt}.csv", index=False)

def split_test_dataset_by_random(cnt, random_state=42):
    df = pd.read_csv("data/combined_data.csv")
    if df.empty:
        print("Empty dataframe cannot be split")
        return
    # Randomly select test set
    np.random.seed(random_state)
    test_indices = np.random.choice(df.index, size=cnt, replace=False)
    test_df = df.loc[test_indices].copy()

    print(f"Choose {cnt} random test data from {test_df['date'].min()} to {test_df['date'].max()}")

    test_df.to_csv(f"data/test/by_random_{cnt}.csv", index=False)

def split_test_dataset_by_symbol_time(symbols, cnt_per_symbol):
    df = pd.read_csv("data/combined_data.csv")
    if df.empty:
        print("Empty dataframe cannot be split")
        return
    # Sort by date for chronological split
    df = df.sort_values('date')

    test_df = pd.DataFrame()
    for symbol in symbols:
        symbol_df = df[df['Symbol'] == symbol]
        cnt = min(len(symbol_df), cnt_per_symbol)
        if cnt == 0:
            print(f"No data for symbol {symbol}")
            continue
        symbol_test_df = symbol_df.iloc[len(symbol_df) - cnt:].copy()
        test_df = pd.concat([test_df, symbol_test_df], ignore_index=True)

    print(f"Choose {cnt_per_symbol} per symbol from {symbols} time based test data from {test_df['date'].min()} to {test_df['date'].max()}")

    test_df.to_csv(f"data/test/by_symbol_time_{cnt_per_symbol}.csv", index=False)

def split_test_dataset_by_symbol_random(symbols, cnt_per_symbol, random_state=42):
    df = pd.read_csv("data/combined_data.csv")
    if df.empty:
        print("Empty dataframe cannot be split")
        return
    # Randomly select test set
    np.random.seed(random_state)
    test_df = pd.DataFrame()
    for symbol in symbols:
        symbol_df = df[df['Symbol'] == symbol]
        cnt = min(len(symbol_df), cnt_per_symbol)
        if cnt == 0:
            print(f"No data for symbol {symbol}")
            continue
        symbol_test_indices = np.random.choice(symbol_df.index, size=cnt, replace=False)
        symbol_test_df = symbol_df.loc[symbol_test_indices].copy()
        test_df = pd.concat([test_df, symbol_test_df], ignore_index=True)

    print(f"Choose {cnt_per_symbol} per symbol from {symbols} random test data from {test_df['date'].min()} to {test_df['date'].max()}")

    test_df.to_csv(f"data/test/by_symbol_random_{cnt_per_symbol}.csv", index=False)

def split_train_dataset():
    df = pd.read_csv("data/combined_data.csv")
    if df.empty:
        print("Empty dataframe cannot be split")
        return
    # exclude test set from training data
    # read every dataset from data/test
    test_files = os.listdir("data/test")
    test_df = pd.DataFrame()
    for file in test_files:
        if file.endswith(".csv"):
            test_df = pd.concat([test_df, pd.read_csv(os.path.join("data/test", file))], ignore_index=True)

    # Remove test data according to the same symbol and date
    df = df[~df.set_index(['Symbol', 'date']).index.isin(test_df.set_index(['Symbol', 'date']).index)]
    # Save the training data
    df.to_csv("data/train.csv", index=False)


def get_sp500_symbols():
    """
    Get the list of ticker symbols for companies in the S&P 500 index
    from Wikipedia.
    
    Returns:
        list: A list of ticker symbols for S&P 500 companies.
    """
    
    try:
        # Get S&P 500 components from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        # Clean symbols to ensure compatibility with yfinance
        symbols = [symbol.replace('.', '-') for symbol in df['Symbol'].tolist()]
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return []
    
def download_dataset(start_date="2015-01-01", end_date="2023-12-31", output_dir="data"):
    """
    Download dataset
    
    Args:
        symbols (list): List of stock ticker symbols
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        output_dir (str): Directory to save the dataset
        
    Returns:
        None
    """
    symbols = get_sp500_symbols()

    # Get stock data
    df = get_stock_data(symbols, start_date, end_date)
    
    if df.empty:
        print("No data to process")
        return
    
    df.to_csv(os.path.join(output_dir, "combined_data.csv"), index=False)


if __name__ == "__main__":
    if not os.path.exists("data/combined_data.csv"):
        download_dataset(start_date="1980-01-01", end_date="2025-03-31", output_dir="data")

    os.makedirs("data/test", exist_ok=True)
    split_test_dataset_by_time(100)
    split_test_dataset_by_random(100)
    split_test_dataset_by_random(1000)

    symbols = ["NVDA", "GOOGL", "AMZN", "AAPL", "MSFT", "META", "TSLA"]
    split_test_dataset_by_symbol_time(symbols, 10)
    split_test_dataset_by_symbol_random(symbols, 10)

    split_train_dataset()
