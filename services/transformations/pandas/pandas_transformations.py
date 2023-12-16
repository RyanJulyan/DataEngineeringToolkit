from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from models.transformations.pandas.pandas_transformations_models import HandleMissingDataColumn


@dataclass
class PandasDataFrameTransformations:
  """
    A class for performing various transformations on a pandas DataFrames in different ways.
    """

  data: pd.DataFrame

  def select_columns(self,
                     columns: List[str]) -> PandasDataFrameTransformations:
    self.data = self.data[columns]
    return self

  def select_rows_by_index(self, index):
    return self.data.loc[index]

  def select_rows_by_condition(self, condition):
    return self.data[condition]

  def handle_missing_values(
      self, column_methods: List[HandleMissingDataColumn]
  ) -> PandasDataFrameTransformations:
    """
        Handle missing data in specified DataFrame columns using specified methods.

        This method iterates over a list of HandleMissingDataColumn instances,
        each specifying a column and how to handle its missing data. The handling
        can be either filling missing data (fillna) or dropping rows with missing data (dropna).

        Args:
            column_methods (List[HandleMissingDataColumn]): A list of HandleMissingDataColumn instances,
                                                           each specifying a column and method for handling missing data.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place.
        """
    for column_method in column_methods:
      if column_method.column_name in self.data.columns:
        if column_method.method == "fillna":
          self.data[column_method.column_name] = self.data[
              column_method.column_name].fillna(column_method.value,
                                                **column_method.kwargs)
        elif column_method.method == "dropna":
          self.data.dropna(subset=[column_method.column_name],
                           inplace=True,
                           **column_method.kwargs)

    return self

  def group_by(self, column):
    return self.data.groupby(column)

  def sort_by_index(self):
    return self.data.sort_index()

  def sort_by_values(self, column):
    return self.data.sort_values(by=column)

  def add_new_column(self, column, values):
    self.data[column] = values
    return self.data

  def drop_column(self, column):
    return self.data.drop(column, axis=1)

  def merge_dataframes(self, other_df, on_column, how='inner'):
    return pd.merge(self.data, other_df, on=on_column, how=how)

  def concat_dataframes(self, other_df):
    return pd.concat([self.data, other_df])

  def create_pivot_table(self, values, index, columns):
    return self.data.pivot_table(values=values, index=index, columns=columns)

  def create_crosstab(self, col1, col2):
    return pd.crosstab(self.data[col1], self.data[col2])

  def apply_function(self, func):
    return self.data.apply(func)

  def string_contains(self, column, substring):
    return self.data[column].str.contains(substring)

  def conditional_operation(self, column, condition, true_val, false_val):
    return np.where(self.data[column] > condition, true_val, false_val)

  # Aggregation methods
  def group_by_and_transform(self, group_column, new_column, agg_func):
    """
      Applies a transformation function to the specified column, partitioning the data 
      by the given group column. This is similar to PARTITION BY in SQL.

      :param group_column: Column to group by.
      :param new_column: Name of the new column to be added with the result.
      :param agg_func: Aggregation function to apply (e.g., 'mean', 'sum', custom function).
      """
    self.data[new_column] = self.data.groupby(
        group_column)[new_column].transform(agg_func)
    return self.data

  def aggregate_data(self, agg_func):
    return self.data.aggregate(agg_func)

  def transform(self, column, agg_func):
    return self.data[column].transform(agg_func)

  # Row operations
  def add_row(self, row_data):
    new_row = pd.DataFrame([row_data], columns=self.data.columns)
    self.data = pd.concat([self.data, new_row], ignore_index=True)
    return self.data

  def modify_row(self, index, row_data):
    self.data.loc[index] = row_data
    return self.data

  def drop_row(self, index):
    self.data = self.data.drop(index)
    return self.data

  # Time series analysis
  def resample_data(self, rule, agg_func='mean'):
    return self.data.resample(rule).agg(agg_func)

  def rolling_window(self, window, agg_func='mean'):
    return self.data.rolling(window=window).agg(agg_func)

  # More advanced column operations
  def arithmetic_operation(self, column, operation, value):
    if operation == 'add':
      self.data[column] += value
    elif operation == 'subtract':
      self.data[column] -= value
    elif operation == 'multiply':
      self.data[column] *= value
    elif operation == 'divide':
      self.data[column] /= value
    return self.data

  # More advanced mapping
  def map_values(self, column, mapping_dict):
    self.data[column] = self.data[column].map(mapping_dict)
    return self.data

  # More complex conditional operations
  def conditional_apply(self, column, condition_func, apply_func):
    self.data.loc[condition_func(self.data),
                  column] = self.data.loc[condition_func(self.data),
                                          column].apply(apply_func)
    return self.data


if __name__ == "__main__":
  # Example Usage
  import numpy as np

  data = {
      'A': [1.23, 2.34, 3.45],
      'B': [4.567672, 5, 6.78],
      'C': ['foo', 'bar', 'baz'],
      'D': ['dog', 'elephant', 'frog'],
      'E': [1, np.nan, 3],
      'F': [4, 5, np.nan],
      "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
  }
  df = pd.DataFrame(data)
  transformations = PandasDataFrameTransformations(df)

  # Apply different missing data handling methods to different columns
  transformations.handle_missing_data([
      HandleMissingDataColumn(column_name='E', method="fillna", value=0),
      HandleMissingDataColumn(column_name='F', method="dropna"),
  ])

  print(transformations.data)
  print()
  print(transformations.data.dtypes)
