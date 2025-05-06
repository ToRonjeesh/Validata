import pandas as pd
import re
import logging
import warnings
from datetime import datetime
from functools import wraps
import time
import os

# =============================================================================
# Module Header: Print current time once as a report header.
# =============================================================================
report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Report generated at: {report_time}\n{'=' * 80}\n")

# =============================================================================
# Configure logging (without repeating the timestamp each time).
# =============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s'
)

# =============================================================================
# Timing decorator to record execution time for key functions.
# =============================================================================
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Starting '{func.__name__}'...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.info(f"Finished '{func.__name__}' in {elapsed:.3f} seconds.")
        return result
    return wrapper

# =============================================================================
# Expectation Suite Runner
# =============================================================================
@timeit
def run_expectation_suite(df: pd.DataFrame, expectations: dict, df_name: str) -> dict:
    """
    Runs an expectation suite on the DataFrame.

    The expectation suite is defined as a dictionary whose keys are column names and values
    are expectation definitions. Each definition can include:
      - "range": a tuple (min, max)
      - "regex": a regex pattern the column's non-null string values must match.
      - "unique": a boolean flag; if True, expect the column values to be unique.
      - "expected_values": a set of expected unique values.

    Returns a dictionary summarizing the results of each expectation.
    """
    results = {}
    logging.info(f"Running expectation suite for DataFrame '{df_name}'...")
    for col, exp in expectations.items():
        col_result = {}
        if col not in df.columns:
            col_result["error"] = f"Column '{col}' not found in DataFrame '{df_name}'."
            results[col] = col_result
            logging.error(col_result["error"])
            continue

        series = df[col]
        # Check range expectation.
        if "range" in exp and pd.api.types.is_numeric_dtype(series):
            low, high = exp["range"]
            below = (series < low).sum()
            above = (series > high).sum()
            passed = (below == 0) and (above == 0)
            col_result["range"] = {
                "expected": f"[{low}, {high}]",
                "failed_below": int(below),
                "failed_above": int(above),
                "passed": passed
            }
            if not passed:
                warnings.warn(
                    f"Expectation failed for column '{col}': values outside the range [{low}, {high}].",
                    RuntimeWarning
                )
        # Check regex expectation.
        if "regex" in exp:
            pattern = re.compile(exp["regex"])
            invalid = series.dropna().apply(lambda x: not bool(pattern.match(str(x))))
            passed = not invalid.any()
            col_result["regex"] = {
                "expected": exp["regex"],
                "failed": int(invalid.sum()),
                "passed": passed
            }
            if not passed:
                warnings.warn(
                    f"Expectation failed for column '{col}': some values do not match the regex {exp['regex']}.",
                    RuntimeWarning
                )
        # Check uniqueness.
        if "unique" in exp and exp["unique"]:
            duplicates = series.duplicated().sum()
            passed = duplicates == 0
            col_result["unique"] = {
                "expected": True,
                "duplicate_count": int(duplicates),
                "passed": passed
            }
            if not passed:
                warnings.warn(
                    f"Expectation failed for column '{col}': found {duplicates} duplicates.",
                    RuntimeWarning
                )
        # Check expected unique values.
        if "expected_values" in exp:
            actual_uniques = set(series.dropna().unique())
            expected_vals = set(exp["expected_values"])
            missing_vals = expected_vals - actual_uniques
            passed = len(missing_vals) == 0
            col_result["expected_values"] = {
                "expected": list(expected_vals),
                "missing": list(missing_vals),
                "passed": passed
            }
            if not passed:
                warnings.warn(
                    f"Expectation failed for column '{col}': missing expected values {missing_vals}.",
                    RuntimeWarning
                )
        results[col] = col_result
    logging.info(f"Expectation suite for '{df_name}' completed.")
    return results

# =============================================================================
# HTML Profiling Report Generator
# =============================================================================
def generate_html_report(df: pd.DataFrame, df_name: str, export_html: str = None) -> str:
    """
    Generates a simple HTML report for the DataFrame profiling.

    The report includes the DataFrame's basic information (shape, memory usage),
    descriptive statistics, value counts for object columns, and the first few rows.

    If export_html is provided, the HTML content is also saved to that file.

    Returns the HTML content as a string.
    """
    html_sections = []
    html_sections.append(f"<h1>DataFrame Profiling Report: {df_name}</h1>")
    html_sections.append(f"<p><b>Shape:</b> {df.shape[0]} rows x {df.shape[1]} columns</p>")
    mem_usage = df.memory_usage(deep=True).sum()
    html_sections.append(f"<p><b>Total Memory Usage:</b> {mem_usage} bytes</p>")

    # Describe numeric columns.
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        html_sections.append("<h2>Numeric Columns Summary</h2>")
        html_sections.append(numeric_cols.describe().to_html(classes="table table-striped", border=0))
    
    # Describe non-numeric columns.
    non_numeric = df.select_dtypes(exclude="number")
    if not non_numeric.empty:
        html_sections.append("<h2>Non-Numeric Columns Summary</h2>")
        html_sections.append(non_numeric.describe(include="all").to_html(classes="table table-striped", border=0))
    
    # Show first 10 rows.
    html_sections.append("<h2>First 10 Rows</h2>")
    html_sections.append(df.head(10).to_html(classes="table table-bordered", border=0))
    
    html_content = "\n".join(html_sections)
    
    if export_html:
        try:
            with open(export_html, "w", encoding="utf-8") as f:
                f.write(html_content)
            logging.info(f"HTML report exported to {export_html}")
        except Exception as e:
            logging.error(f"Error exporting HTML report: {e}")
    
    return html_content

# =============================================================================
# DataFrame Validation Function with All Enhanced Features
# (This version now optionally runs an expectation suite and can produce an HTML report.)
# =============================================================================
@timeit
def validate_df_enhanced(
    df: pd.DataFrame,
    df_name: str,
    expected_columns: list = None,
    expected_dtypes: dict = None,
    min_rows: int = None,
    range_checks: dict = None,
    pattern_validations: dict = None,
    expected_unique: dict = None,
    custom_validators: list = None,
    expectations: dict = None,
    export_html: str = None
) -> None:
    """
    Validates and profiles a DataFrame with additional features.
    
    In addition to the standard validations, this function can run an expectation suite and 
    generate an HTML report.
    
    Parameters:
      (All parameters as in validate_df, plus:)
      expectations (dict, optional): Dictionary defining an expectation suite.
      export_html (str, optional): Filename to export an HTML report.
    """
    validate_df(df, df_name, expected_columns, expected_dtypes, min_rows,
                range_checks, pattern_validations, expected_unique, custom_validators)
    
    # If an expectation suite is provided, run it.
    if expectations:
        exp_results = run_expectation_suite(df, expectations, df_name)
        logging.info(f"Expectation results for '{df_name}': {exp_results}")
    
    # If HTML export is requested, generate and save the report.
    if export_html:
        html_report = generate_html_report(df, df_name, export_html)
        logging.info(f"HTML report for '{df_name}' generated.")

# =============================================================================
# Enhanced Join Validation Function (supporting all join types)
# =============================================================================
@timeit
def validate_join(left: pd.DataFrame, right: pd.DataFrame, on: list, join_type: str = 'inner') -> None:
    """
    Validates join keys between two DataFrames with enhanced checks.
    
    Checks include:
      - Existence of join keys in both DataFrames.
      - Ensuring no null values in join key columns.
      - No duplicate composite keys among rows with complete join keys.
      - For asymmetric joins, warns if composite keys are missing in the other DataFrame.
      
    Composite keys are created using a vectorized string join.
    
    Parameters:
      left (pd.DataFrame): Left DataFrame.
      right (pd.DataFrame): Right DataFrame.
      on (list): List of join key columns.
      join_type (str): The join type (e.g., "inner", "left", "right", "outer").
    """
    if not on:
        logging.error("Join key list is empty. At least one join column must be specified.")
        return

    logging.info("Starting join validation")
    try:
        for key in on:
            if key not in left.columns:
                logging.error(f"Join key '{key}' not found in left DataFrame.")
            else:
                left_nulls = left[key].isnull().sum()
                if left_nulls:
                    warnings.warn(f"Left DataFrame join key '{key}' has {left_nulls} null value(s).", RuntimeWarning)
            if key not in right.columns:
                logging.error(f"Join key '{key}' not found in right DataFrame.")
            else:
                right_nulls = right[key].isnull().sum()
                if right_nulls:
                    warnings.warn(f"Right DataFrame join key '{key}' has {right_nulls} null value(s).", RuntimeWarning)
                    
        left_missing = left[on].isnull().any(axis=1)
        right_missing = right[on].isnull().any(axis=1)
        if left_missing.any():
            warnings.warn(f"Left DataFrame has {left_missing.sum()} row(s) with missing join key values in columns: {on}.", RuntimeWarning)
        if right_missing.any():
            warnings.warn(f"Right DataFrame has {right_missing.sum()} row(s) with missing join key values in columns: {on}.", RuntimeWarning)
        
        left_complete = left.dropna(subset=on)
        right_complete = right.dropna(subset=on)
        left_keys = left_complete[on].astype(str).agg("||".join, axis=1)
        right_keys = right_complete[on].astype(str).agg("||".join, axis=1)
        
        left_dup = left_keys.duplicated().sum()
        right_dup = right_keys.duplicated().sum()
        if left_dup:
            warnings.warn(f"Left DataFrame (after dropping rows with null join keys) has {left_dup} duplicate composite key(s) for join columns {on}.", RuntimeWarning)
        if right_dup:
            warnings.warn(f"Right DataFrame (after dropping rows with null join keys) has {right_dup} duplicate composite key(s) for join columns {on}.", RuntimeWarning)
        
        # For asymmetric joins, check composite key differences.
        if join_type in ['left', 'outer', 'full'] or join_type == 'right':
            left_set = set(left_keys)
            right_set = set(right_keys)
            if join_type in ['left', 'outer', 'full']:
                missing_in_right = left_set - right_set
                if missing_in_right:
                    warnings.warn(f"For join key columns {on}, composite key(s) missing in right DataFrame: {missing_in_right}", RuntimeWarning)
            if join_type in ['right', 'outer', 'full']:
                missing_in_left = right_set - left_set
                if missing_in_left:
                    warnings.warn(f"For join key columns {on}, composite key(s) missing in left DataFrame: {missing_in_left}", RuntimeWarning)
                    
        logging.info("Join validation completed successfully.")
    except Exception as e:
        logging.error(f"Error during join validation: {e}")

# =============================================================================
# Enhanced Safe Join Function (calls enhanced validate_join)
# =============================================================================
@timeit
def safe_join(left: pd.DataFrame, right: pd.DataFrame, on: list, join_type: str = 'inner') -> pd.DataFrame:
    """
    Safely merges two DataFrames after performing enhanced join validations.
    
    The merge is performed using pandas.merge with an indicator column (which is later removed).
    If the merge fails, the function logs an error and returns None.
    
    Parameters:
      left (pd.DataFrame): Left DataFrame.
      right (pd.DataFrame): Right DataFrame.
      on (list): List of join key columns.
      join_type (str): Type of join (default "inner").
      
    Returns:
      pd.DataFrame: The merged DataFrame if successful; otherwise, None.
    """
    logging.info("Starting safe join operation.")
    validate_join(left, right, on, join_type)
    try:
        merged = left.merge(right, on=on, how=join_type, indicator=True)
        logging.info("Join successful. Merge indicator summary:")
        logging.info(merged['_merge'].value_counts().to_string())
        merged.drop(columns=['_merge'], inplace=True)
        return merged
    except Exception as e:
        logging.error(f"Error during join operation: {e}")
        return None

# =============================================================================
# Enhanced Notebook Testing Wrapper Function with Optional HTML Export
# =============================================================================
@timeit
def test_notebook_dataframes(*dfs, df_names=None, join_on=None, join_type='inner',
                             range_checks=None, pattern_validations=None,
                             expected_unique=None, custom_validators=None,
                             expectations=None, export_file=None, export_html=None):
    """
    Wrapper function to test one or multiple DataFrames before production.
    
    - For a single DataFrame, it generates a complete profiling report.
    - For 2–4 DataFrames, each is profiled and then merged sequentially using safe_join.
    - Optionally runs an expectation suite and generates an HTML report.
    
    Additional optional validations include range_checks, pattern_validations, expected_unique,
    custom_validators, and an expectation suite.
    
    If export_file is provided, log output is saved to that file.
    If export_html is provided, an HTML report is generated.
    
    Parameters:
      *dfs: 1 to 4 DataFrame objects.
      df_names (list, optional): Names for each DataFrame.
      join_on (list, optional): List of join key columns (required if merging multiple DataFrames).
      join_type (str, optional): Type of join (default "inner").
      range_checks, pattern_validations, expected_unique, custom_validators:
            Additional per-column validations.
      expectations (dict, optional): Expectation suite (see run_expectation_suite).
      export_file (str, optional): Filename to export the log output.
      export_html (str, optional): Filename to export an HTML profiling report.
      
    Returns:
      pd.DataFrame: The merged DataFrame if more than one DataFrame is provided; otherwise, None.
    """
    num_dfs = len(dfs)
    if num_dfs == 0 or num_dfs > 4:
        logging.error("Please provide between 1 and 4 DataFrame objects.")
        return None

    if not df_names or len(df_names) != num_dfs:
        df_names = [f"DF{i+1}" for i in range(num_dfs)]
    
    if export_file:
        file_handler = logging.FileHandler(export_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    if num_dfs == 1:
        logging.info("Single DataFrame provided. Generating full profiling report.")
        validate_df_enhanced(dfs[0], df_names[0], expected_columns=None, expected_dtypes=None,
                             min_rows=None, range_checks=range_checks,
                             pattern_validations=pattern_validations,
                             expected_unique=expected_unique,
                             custom_validators=custom_validators,
                             expectations=expectations, export_html=export_html)
        return None
    
    for df, name in zip(dfs, df_names):
        logging.info(f"Profiling DataFrame '{name}':")
        validate_df_enhanced(df, name, range_checks=range_checks,
                             pattern_validations=pattern_validations,
                             expected_unique=expected_unique,
                             custom_validators=custom_validators,
                             expectations=expectations, export_html=None)
    
    if join_on is None or not join_on:
        logging.error("Join keys must be provided for merging multiple DataFrames.")
        return None
    
    logging.info("Starting sequential safe joins.")
    merged_df = dfs[0]
    current_name = df_names[0]
    for i in range(1, num_dfs):
        next_df = dfs[i]
        next_name = df_names[i]
        logging.info(f"Joining '{current_name}' with '{next_name}' on keys: {join_on}")
        merged_df = safe_join(merged_df, next_df, join_on, join_type)
        if merged_df is None:
            logging.error(f"Join between '{current_name}' and '{next_name}' failed. Aborting further joins.")
            return None
        current_name = f"{current_name}_{next_name}"
    logging.info("All joins completed successfully.")
    logging.info("Final merged DataFrame profiling:")
    validate_df_enhanced(merged_df, "MergedDF", range_checks=range_checks,
                         pattern_validations=pattern_validations,
                         expected_unique=expected_unique,
                         custom_validators=custom_validators,
                         expectations=expectations, export_html=export_html)
    
    return merged_df
