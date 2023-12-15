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

  def handle_missing_data(
      self, column_methods: List[HandleMissingDataColumn]) -> None:
    """
        Handle missing data in specified DataFrame columns using specified methods.

        This method iterates over a list of HandleMissingDataColumn instances,
        each specifying a column and how to handle its missing data. The handling
        can be either filling missing data (fillna) or dropping rows with missing data (dropna).

        Args:
            column_methods (List[HandleMissingDataColumn]): A list of HandleMissingDataColumn instances,
                                                           each specifying a column and method for handling missing data.

        Returns:
            None: The method modifies the DataFrame in place.
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
