#!/usr/bin/env python3
"""
Script to verify that all remaining stock folders have data up to the latest trading date (2025-10-09)
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

def check_stock_data_completeness():
    """Check if all stock folders have data up to the latest trading date"""
    stock_data_dir = Path("stock_data")
    latest_date = "2025-10-09"
    latest_date_dt = pd.to_datetime(latest_date)

    total_folders = 0
    complete_folders = 0
    incomplete_folders = 0
    missing_files = 0
    error_files = 0

    print("Checking data completeness for all remaining stock folders...")
    print(f"Target date: {latest_date}")
    print("-" * 60)

    # Iterate through all stock folders
    for stock_folder in stock_data_dir.iterdir():
        if not stock_folder.is_dir():
            continue

        total_folders += 1
        ticker = stock_folder.name
        csv_file = stock_folder / f"{ticker}_price.csv"

        if not csv_file.exists():
            missing_files += 1
            print(f"❌ {ticker}: Missing CSV file")
            continue

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            if df.empty:
                incomplete_folders += 1
                print(f"❌ {ticker}: Empty CSV file")
                continue

            # Check if Date column exists
            if 'Date' not in df.columns:
                # Try to find date column
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                if date_cols:
                    date_col = date_cols[0]
                else:
                    error_files += 1
                    print(f"❌ {ticker}: No Date column found")
                    continue
            else:
                date_col = 'Date'

            # Convert dates and find latest date
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])

            if df.empty:
                incomplete_folders += 1
                print(f"❌ {ticker}: No valid dates")
                continue

            latest_data_date = df[date_col].max()
            days_diff = (latest_date_dt - latest_data_date).days

            if days_diff <= 2:  # Has data within 2 days of target (accounting for weekends/delays)
                complete_folders += 1
            else:
                incomplete_folders += 1
                print(f"⚠️  {ticker}: Latest data is {latest_data_date.strftime('%Y-%m-%d')} ({days_diff} days old)")

        except Exception as e:
            error_files += 1
            print(f"❌ {ticker}: Error reading file - {str(e)[:50]}...")

        # Progress indicator every 1000 files
        if total_folders % 1000 == 0:
            print(f"Processed {total_folders} files...")

    print("\n" + "=" * 60)
    print("COMPLETENESS CHECK RESULTS:")
    print("=" * 60)
    print(f"Total folders checked: {total_folders}")
    print(f"Complete (up to date): {complete_folders}")
    print(f"Incomplete (outdated): {incomplete_folders}")
    print(f"Missing files: {missing_files}")
    print(f"Error files: {error_files}")
    print()

    completeness_rate = (complete_folders / total_folders * 100) if total_folders > 0 else 0
    print(f"Completeness rate: {completeness_rate:.1f}%")

    if completeness_rate >= 95:
        print("✅ Excellent! Data is highly complete.")
    elif completeness_rate >= 80:
        print("⚠️  Good, but some stocks need updating.")
    else:
        print("❌ Many stocks are missing recent data.")

if __name__ == "__main__":
    check_stock_data_completeness()