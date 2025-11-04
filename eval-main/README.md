### Dataset
Earning records and prices on earning days and post-earning days for stocks in the current S&P 500 from 1980 to 2025

#### Test dataset list
- data/test/by_symbol_random_10.csv

  random choose 10 records for every stock in ["NVDA", "GOOGL", "AMZN", "AAPL", "MSFT", "META", "TSLA"]
- data/test/by_symbol_time_10.csv

  latest 10 records for every stock in ["NVDA", "GOOGL", "AMZN", "AAPL", "MSFT", "META", "TSLA"]
- data/test/by_random_100.csv

  random choose 100 records
- data/test/by_time_100.csv

  latest 100 records
- data/test/by_random_1000.csv

  random choose 1000 records

### Evaluate other methods
1. Choose one dataset from data/test, be careful not to mix the answers (`Post Open` and `Open Diff(%)`) to input for other methods.
2. Then modify output in data/predictions according to the results by other methods
3. Run the command
```
python evaluate.py
```

### Train gradient boost model and predict
This will overwrite data/predictions
```
python gradient_boost.py
```

### Regenerate dataset
```
python dataset.py
```
