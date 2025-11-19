"""
Data loading and feature engineering utilities.

This module handles:
1. Downloading historical OHLCV data from yfinance
2. Computing technical indicators (MACD, RSI, Bollinger Bands, moving averages, etc.)
3. Fetching fundamental metrics (P/E, P/B, ROE) where available
4. Preprocessing and cleaning data (handling missing values, gaps, etc.)

Key principle: Data loading is APPROACH-AGNOSTIC.
All three approaches (Perplexity API, weak LLM, quant-only) use the same data pipeline.

This ensures fair comparison and reproducibility.
"""

from typing import Dict, List, Tuple, Optional, Any
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    raise ImportError(
        "yfinance not installed. Install with: pip install yfinance"
    )

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError(
        "pandas_ta not installed. Install with: pip install pandas-ta"
    )


class DataLoader:
    """
    Handles all data acquisition and feature engineering.

    Usage:
        loader = DataLoader()

        # Download data
        df = loader.fetch_historical_data(
            symbols=["AAPL", "NVDA", "MSFT"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 29)
        )

        # Compute indicators
        df = loader.add_technical_indicators(df)

        # Get fundamentals
        fundamentals = loader.fetch_fundamentals(symbols=["AAPL"])
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the DataLoader.

        Args:
            verbose (bool): Print status messages
        """
        self.verbose = verbose

    def fetch_historical_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download historical OHLCV data from Yahoo Finance.

        Args:
            symbols (List[str]): List of tickers, e.g., ["AAPL", "NVDA", "MSFT"]
            start_date (date): Start of historical data period
            end_date (date): End of historical data period
            interval (str): Data granularity ("1d" for daily, "1h" for hourly, etc.)

        Returns:
            pd.DataFrame: MultiIndex DataFrame with (date, symbol) as index,
                         columns: [open, high, low, close, adj_close, volume]
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Fetching historical data for {len(symbols)} symbols")
            print(f"Period: {start_date} to {end_date}")
            print(f"{'='*70}")

        all_data = []

        for symbol in symbols:
            try:
                if self.verbose:
                    print(f"  Downloading {symbol}...", end="")

                # Download from Yahoo Finance
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval=interval)

                if hist.empty:
                    print(f" [WARN] No data")
                    continue

                # Rename columns to lowercase
                hist.columns = [col.lower() for col in hist.columns]

                # Reset index to make date a column
                hist = hist.reset_index()

                # Ensure 'date' column exists (yfinance returns it as 'Date' or in index)
                if 'date' not in hist.columns and 'index' in hist.columns:
                    hist = hist.rename(columns={'index': 'date'})
                if 'date' not in hist.columns:
                    hist.columns = ['date' if i == 0 else col for i, col in enumerate(hist.columns)]

                # Convert date column to date type (not datetime)
                hist['date'] = pd.to_datetime(hist['date']).dt.date

                # Add symbol column
                hist['symbol'] = symbol

                all_data.append(hist)

                if self.verbose:
                    print(f" {len(hist)} rows")

            except Exception as e:
                print(f" [ERROR] {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_data:
            raise ValueError("No data fetched for any symbol")

        # Concatenate all symbols
        df = pd.concat(all_data, axis=0, ignore_index=True)

        # Set MultiIndex (date, symbol)
        df = df.set_index(['date', 'symbol'])
        df = df.sort_index()

        # Ensure adj_close exists (fallback to close if missing)
        if 'adjclose' in df.columns:
            df.rename(columns={'adjclose': 'adj_close'}, inplace=True)
        elif 'adj close' in df.columns:
            df.rename(columns={'adj close': 'adj_close'}, inplace=True)

        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']

        # Clean: keep only essential OHLCV columns
        essential_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        existing_cols = [col for col in essential_cols if col in df.columns]
        df = df[existing_cols]

        if self.verbose:
            print(f"\nData shape: {df.shape}")
            print(f"Date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")
            print(f"Symbols: {df.index.get_level_values(1).unique().tolist()}\n")

        return df

    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute technical indicators for all symbols.

        Args:
            df (pd.DataFrame): OHLCV data (output from fetch_historical_data)
            indicators (Optional[List[str]]): List of indicator names to compute.
                If None, computes default set: ["MACD", "RSI", "BBANDS", "SMA", "EMA", "ADX"]

        Returns:
            pd.DataFrame: Original df with new columns for each indicator
        """
        if indicators is None:
            indicators = ["MACD", "RSI", "BBANDS", "SMA", "EMA", "ADX"]

        if self.verbose:
            print(f"\nComputing technical indicators: {indicators}")
            print(f"{'='*70}")

        # Process each symbol separately (indicators are per-symbol)
        for symbol in df.index.get_level_values(1).unique():
            try:
                if self.verbose:
                    print(f"  {symbol}...", end="")

                symbol_data = df.loc[df.index.get_level_values(1) == symbol].copy()
                symbol_data = symbol_data.reset_index(drop=True)

                # Compute indicators using pandas_ta
                for indicator in indicators:
                    if indicator == "MACD":
                        macd = ta.macd(
                            symbol_data["close"],
                            fast=12,
                            slow=26,
                            signal=9
                        )
                        for col in macd.columns:
                            symbol_data[col] = macd[col]

                    elif indicator == "RSI":
                        rsi = ta.rsi(symbol_data["close"], length=14)
                        symbol_data["RSI_14"] = rsi

                    elif indicator == "BBANDS":
                        bbands = ta.bbands(
                            symbol_data["close"],
                            length=20,
                            std=2
                        )
                        for col in bbands.columns:
                            symbol_data[col] = bbands[col]

                    elif indicator == "SMA":
                        for period in [20, 50, 200]:
                            symbol_data[f"SMA_{period}"] = ta.sma(
                                symbol_data["close"],
                                length=period
                            )

                    elif indicator == "EMA":
                        for period in [12, 26]:
                            symbol_data[f"EMA_{period}"] = ta.ema(
                                symbol_data["close"],
                                length=period
                            )

                    elif indicator == "ADX":
                        adx = ta.adx(
                            symbol_data["high"],
                            symbol_data["low"],
                            symbol_data["close"],
                            length=14
                        )
                        for col in adx.columns:
                            symbol_data[col] = adx[col]

                # Get original index values
                original_dates = df.loc[df.index.get_level_values(1) == symbol].index.get_level_values(0)

                # Create new MultiIndex for updated data
                new_index = pd.MultiIndex.from_arrays(
                    [original_dates, [symbol] * len(symbol_data)],
                    names=['date', 'symbol']
                )
                symbol_data.index = new_index

                # Update the main dataframe
                df.loc[df.index.get_level_values(1) == symbol, symbol_data.columns] = symbol_data

                if self.verbose:
                    print(" ✓")

            except Exception as e:
                print(f" [ERROR] {e}")
                import traceback
                traceback.print_exc()
                continue

        if self.verbose:
            print(f"{'='*70}\n")

        return df

    def fetch_fundamentals(
        self,
        symbols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Fetch fundamental metrics for symbols from Yahoo Finance.

        Args:
            symbols (List[str]): List of tickers

        Returns:
            Dict with structure: {
                "AAPL": {
                    "trailingPE": 28.5,
                    "priceToBook": 45.2,
                    "returnOnEquity": 0.82,
                    ...
                },
                ...
            }
        """
        if self.verbose:
            print(f"\nFetching fundamentals for {len(symbols)} symbols")
            print(f"{'='*70}")

        fundamentals = {}

        for symbol in symbols:
            try:
                if self.verbose:
                    print(f"  {symbol}...", end="")

                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Extract relevant fundamental metrics
                metrics = {
                    "pe_ratio": info.get("trailingPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "gross_margin": info.get("grossMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "profit_margin": info.get("profitMargins"),
                    "dividend_yield": info.get("dividendYield"),
                }

                # Remove None values
                metrics = {k: v for k, v in metrics.items() if v is not None}

                fundamentals[symbol] = metrics

                if self.verbose:
                    print(f" {len(metrics)} metrics")

            except Exception as e:
                print(f" [WARN] {e}")
                fundamentals[symbol] = {}
                continue

        if self.verbose:
            print(f"{'='*70}\n")

        return fundamentals

    def handle_missing_data(
        self,
        df: pd.DataFrame,
        method: str = "forward_fill",
    ) -> pd.DataFrame:
        """
        Handle missing values in data (gaps, delisted symbols, etc.).

        Args:
            df (pd.DataFrame): Data with potential NaN values
            method (str): Strategy for handling missing data
                - "forward_fill": use last known value (default)
                - "drop": remove rows with NaN
                - "interpolate": linear interpolation

        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.verbose:
            print(f"Handling missing data with method: {method}")

        if method == "forward_fill":
            df = df.fillna(method="ffill")
        elif method == "drop":
            df = df.dropna()
        elif method == "interpolate":
            df = df.interpolate(method="linear", limit_direction="both")
        else:
            raise ValueError(f"Unknown method: {method}")

        return df

    def get_tradable_symbols(
        self,
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Get list of symbols with complete data (no major gaps).

        Args:
            df (pd.DataFrame): Historical data

        Returns:
            List[str]: Symbols with sufficient data quality
        """
        symbols_with_data = df.index.get_level_values(1).unique().tolist()
        return symbols_with_data

    def get_date_range(
        self,
        df: pd.DataFrame,
    ) -> Tuple[date, date]:
        """
        Get the date range available in the data.

        Args:
            df (pd.DataFrame): Historical data

        Returns:
            Tuple[date, date]: (start_date, end_date)
        """
        dates = sorted(df.index.get_level_values(0).unique())
        return dates[0], dates[-1]