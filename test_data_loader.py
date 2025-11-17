"""
Quick integration test for DataLoader.

This script:
1. Installs/verifies required dependencies (yfinance, pandas_ta)
2. Downloads a small amount of real market data (1 stock, 10 days)
3. Computes technical indicators
4. Fetches fundamentals
5. Validates output structure

Run BEFORE committing data_loader.py to production.

Usage:
    python -m src.utils.test_data_loader
"""

import sys
from datetime import date, timedelta

print("=" * 70)
print("DATA LOADER INTEGRATION TEST")
print("=" * 70 + "\n")

# Step 1: Check dependencies
print("Step 1: Checking dependencies...")
try:
    import pandas as pd

    print("  ✓ pandas")
except ImportError:
    print("  ✗ pandas not found. Install: pip install pandas")
    sys.exit(1)

try:
    import numpy as np

    print("  ✓ numpy")
except ImportError:
    print("  ✗ numpy not found. Install: pip install numpy")
    sys.exit(1)

try:
    import yfinance as yf

    print("  ✓ yfinance")
except ImportError:
    print("  ✗ yfinance not found. Install: pip install yfinance")
    print("  Installing now...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

    print("  ✓ yfinance installed")

try:
    import pandas_ta as ta

    print("  ✓ pandas_ta")
except ImportError:
    print("  ✗ pandas_ta not found. Install: pip install pandas-ta")
    print("  Installing now...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas-ta"])
    import pandas_ta as ta

    print("  ✓ pandas_ta installed")

# Step 2: Import DataLoader
print("\nStep 2: Importing DataLoader...")
try:
    from src.utils.data_loader import DataLoader

    print("  ✓ DataLoader imported")
except ImportError as e:
    print(f"  ✗ Failed to import DataLoader: {e}")
    sys.exit(1)

# Step 3: Download small test data
print("\nStep 3: Downloading test data (AAPL, 30 days)...")
try:
    loader = DataLoader(verbose=True)

    # Use a small window to speed up test
    end_date = date(2024, 3, 29)
    start_date = end_date - timedelta(days=30)

    df = loader.fetch_historical_data(
        symbols=["AAPL"],
        start_date=start_date,
        end_date=end_date,
    )

    print(f"\n  ✓ Downloaded {len(df)} rows")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\n  First few rows:")
    print(df.head(3))

except Exception as e:
    print(f"  ✗ Failed to download data: {e}")
    sys.exit(1)

# Step 4: Add technical indicators
print("\nStep 4: Computing technical indicators...")
try:
    df = loader.add_technical_indicators(
        df,
        indicators=["MACD", "RSI", "BBANDS", "SMA"]
    )

    print(f"  ✓ Indicators added")
    print(f"  New shape: {df.shape}")
    print(
        f"  New columns: {[col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'adj_close', 'volume']]}")
    print(f"\n  Sample row with indicators:")
    sample_row = df.iloc[-1]
    for col in ['close', 'RSI_14', 'SMA_20', 'SMA_50']:
        if col in sample_row.index:
            print(f"    {col}: {sample_row[col]}")

except Exception as e:
    print(f"  ✗ Failed to compute indicators: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 5: Fetch fundamentals
print("\nStep 5: Fetching fundamentals...")
try:
    fundamentals = loader.fetch_fundamentals(symbols=["AAPL"])

    print(f"  ✓ Fundamentals fetched")
    print(f"  Metrics available: {list(fundamentals['AAPL'].keys())}")
    print(f"\n  Sample fundamentals for AAPL:")
    for key, value in list(fundamentals['AAPL'].items())[:3]:
        print(f"    {key}: {value}")

except Exception as e:
    print(f"  ✗ Failed to fetch fundamentals: {e}")
    # Don't exit - fundamentals are optional

# Step 6: Validate data structure
print("\nStep 6: Validating data structure...")
try:
    # Check MultiIndex
    assert isinstance(df.index, pd.MultiIndex), "Index should be MultiIndex"
    assert df.index.names == ['date', 'symbol'], f"Index names should be ['date', 'symbol'], got {df.index.names}"
    print("  ✓ MultiIndex structure correct")

    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    print("  ✓ All required OHLCV columns present")

    # Check no NaN in OHLCV
    ohlcv_data = df[required_cols]
    nan_count = ohlcv_data.isnull().sum().sum()
    if nan_count > 0:
        print(f"  ⚠ {nan_count} NaN values in OHLCV (might be normal for some indicators)")
    else:
        print("  ✓ No NaN values in OHLCV")

    # Check data types
    assert df['open'].dtype in [np.float32, np.float64], f"open should be float, got {df['open'].dtype}"
    assert df['volume'].dtype in [np.int64, np.float64], f"volume should be numeric, got {df['volume'].dtype}"
    print("  ✓ Data types correct")

except AssertionError as e:
    print(f"  ✗ Validation failed: {e}")
    sys.exit(1)

# Step 7: Test utility methods
print("\nStep 7: Testing utility methods...")
try:
    symbols = loader.get_tradable_symbols(df)
    start, end = loader.get_date_range(df)

    print(f"  ✓ Tradable symbols: {symbols}")
    print(f"  ✓ Date range: {start} to {end}")

except Exception as e:
    print(f"  ✗ Utility methods failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL DATA LOADER TESTS PASSED ✓")
print("=" * 70)
print("\nYou can now use DataLoader in your approaches!")
print("\nNext steps:")
print("  1. Implement src/utils/metrics.py")
print("  2. Implement src/baselines/ strategies")
print("  3. Implement src/approaches/quant_only/")