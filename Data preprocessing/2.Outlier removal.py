import pandas as pd
import numpy as np

def remove_outliers_sliding_window(df, column_name, window_size=10, upper_multiplier=5.0, lower_multiplier=0.2):
    """
    Removes outliers from a specific column of a DataFrame based on a sliding window approach.

    An outlier is defined as a value that is greater than `upper_multiplier` times
    or less than `lower_multiplier` times the median of the preceding `window_size` data points.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to clean.
        window_size (int): The number of preceding data points to consider for the rolling median.
        upper_multiplier (float): The multiplier for the upper threshold.
        lower_multiplier (float): The multiplier for the lower threshold.

    Returns:
        pd.DataFrame: A DataFrame with outliers removed.
    """
    print(f"--- Applying Sliding Window Outlier Removal on '{column_name}' ---")
    print(f"Window Size: {window_size}, Upper Multiplier: {upper_multiplier}, Lower Multiplier: {lower_multiplier}")

    # Calculate the rolling median. `min_periods=1` allows calculation even at the beginning of the series.
    # `closed='left'` ensures that the current point is not included in the window for its own calculation.
    rolling_median = df[column_name].rolling(window=window_size, min_periods=1, closed='left').median()

    # Define the dynamic upper and lower thresholds based on the rolling median
    upper_threshold = rolling_median * upper_multiplier
    lower_threshold = rolling_median * lower_multiplier

    # Identify outliers. We use .copy() to avoid SettingWithCopyWarning.
    df_copy = df.copy()
    df_copy['is_outlier'] = ~df_copy[column_name].between(lower_threshold, upper_threshold)

    # The first few points might not have a stable window, handle them carefully.
    # For simplicity, we can ignore the very first point if its window is not full.
    # Or, as done here with min_periods=1, the window grows, which is a reasonable approach.
    
    # Filter out the outliers
    df_cleaned = df_copy[~df_copy['is_outlier']].drop(columns=['is_outlier'])
    
    original_rows = len(df)
    cleaned_rows = len(df_cleaned)
    removed_count = original_rows - cleaned_rows
    
    print(f"Removed {removed_count} outlier rows ({ (removed_count / original_rows) * 100:.2f}% of data).")
    print("--------------------------------------------------")
    
    return df_cleaned