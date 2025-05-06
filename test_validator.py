# tests/test_validator.py

import unittest
import pandas as pd
import os
from data_validator.validator import (
    validate_df,
    run_expectation_suite,
    validate_join,
    safe_join,
    test_notebook_dataframes,
    generate_html_report
)
import warnings

# Suppress warnings during testing if desired.
warnings.simplefilter("ignore", category=RuntimeWarning)


class TestDataValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create sample DataFrames for testing.
        cls.df1 = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': ['A', 'B', 'C', 'D'],
            'col3': [10, 20, 30, 40],
            'col4': ['X', 'Y', 'Z', 'W'],
            'value1': [100, 200, 300, 400]
        })

        cls.df2 = pd.DataFrame({
            'col1': [3, 4, 5, 6],
            'col2': ['C', 'D', 'E', 'F'],
            'col3': [30, 40, 50, 60],
            'col4': ['X', 'W', 'V', 'U'],
            'value2': [1000, 2000, 3000, 4000]
        })

        # An expectation suite similar to a Great Expectations suite.
        cls.expectations = {
            'col1': {'range': (1, 6), 'unique': True},
            'col2': {'regex': r'^[A-Z]$'},
            'col3': {'range': (5, 100)},
            'col4': {'expected_values': {'X', 'Y', 'Z', 'W'}}
        }

    def test_validate_df(self):
        # Test the validation function to ensure it does not raise errors.
        # This function prints output through logging.
        validate_df(self.df1, "df1")
        self.assertEqual(self.df1.shape, (4, 5))

    def test_run_expectation_suite(self):
        # Run expectation suite on df1 and check results.
        results = run_expectation_suite(self.df1, self.expectations, "df1")
        # For col1, it is expected to be unique and within the range.
        self.assertIn("col1", results)
        self.assertTrue(results["col1"]["range"]["passed"])
        self.assertTrue(results["col1"]["unique"]["passed"])
        # For col2, check regex expectation.
        self.assertIn("col2", results)
        self.assertTrue(results["col2"]["regex"]["passed"])

    def test_validate_join_and_safe_join(self):
        # Test join validation and safe join
        # This should warn if there are issues but still perform an inner join.
        try:
            validate_join(self.df1, self.df2, on=["col1", "col2", "col3", "col4"], join_type='inner')
        except Exception as e:
            self.fail(f"validate_join raised an exception: {e}")

        merged = safe_join(self.df1, self.df2, on=["col1", "col2", "col3", "col4"], join_type='inner')
        # The merge based on our sample data should produce at least one matching row.
        self.assertIsNotNone(merged)
        self.assertGreaterEqual(merged.shape[0], 1)

    def test_html_report_generation(self):
        # Test that the HTML report generator returns a non-empty string.
        html_content = generate_html_report(self.df1, "df1_test", export_html="temp_report.html")
        self.assertTrue(isinstance(html_content, str))
        self.assertGreater(len(html_content), 0)
        # Check that the file was created and then remove it.
        self.assertTrue(os.path.exists("temp_report.html"))
        os.remove("temp_report.html")

    def test_test_notebook_dataframes_wrapper(self):
        # Test the wrapper function for multiple DataFrames. This should return a merged DataFrame.
        merged_df = test_notebook_dataframes(
            self.df1, self.df2,
            df_names=["df1", "df2"],
            join_on=["col1", "col2", "col3", "col4"],
            join_type='inner',
            expectations=self.expectations,
            export_file=None,
            export_html=None
        )
        self.assertIsNotNone(merged_df)
        # The merged DataFrame should have columns from both DataFrames.
        self.assertIn("value1", merged_df.columns)
        self.assertIn("value2", merged_df.columns)


if __name__ == '__main__':
    unittest.main()
