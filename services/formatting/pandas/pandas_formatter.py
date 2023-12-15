from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import pandas as pd

from models.formatting.pandas.pandas_formatter_models import (
    ConvertColumnToType,
    RoundColumnDecimalPlaces,
    FormatFloatColumn,
    FormatStringColumn,
    RenameColumn,
    ConvertToDateTime,
    FormatDateTimeColumn,
    SortDetails,
)
from services.formatting.i_formatter import IFormatter


@dataclass
class PandasDataFrameFormatter(IFormatter):
  """Class for formatting a pandas DataFrame in various ways."""

  data: pd.DataFrame

  def set_display_options(self,
                          max_rows: Optional[int] = None,
                          max_columns: Optional[int] = None,
                          precision: Optional[int] = None) -> None:
    """
        Set display options for the DataFrame.

        Args:
            max_rows (Optional[int]): Maximum number of rows to display.
            max_columns (Optional[int]): Maximum number of columns to display.
            precision (Optional[int]): Decimal precision for displaying floats.
        """
    if max_rows is not None:
      pd.set_option("display.max_rows", max_rows)
    if max_columns is not None:
      pd.set_option("display.max_columns", max_columns)
    if precision is not None:
      pd.set_option("display.precision", precision)

  def apply_style(self, styler_function: Callable) -> Any:
    """
        Apply a styling function to the DataFrame.

        Args:
            styler_function (Callable): A function to apply for styling.

        Returns:
            pd.io.formats.style.Styler: The styled DataFrame.
        """
    return self.data.style.apply(styler_function)

  def convert_type(self, column_details: List[ConvertColumnToType]) -> None:
    """
        Convert the data types of specified columns.

        Args:
            column_details (List[ConvertColumnToType]): A list of ConvertColumnToType instances specifying column and type to convert to.
        """
    for column in column_details:
      if column.column_name in self.data.columns:
        self.data[column.column_name] = self.data[column.column_name].astype(
            column.dtype_name)

  def round_value(self,
                  column_details: List[RoundColumnDecimalPlaces]) -> None:
    """
        Round the values in specified columns to a set number of decimal places.

        Args:
            column_details (List[RoundColumnDecimalPlaces]): A list of RoundColumnDecimalPlaces instances specifying column and the number of decimal places.
        """
    for column in column_details:
      if column.column_name in self.data.columns:
        self.data[column.column_name] = self.data[column.column_name].round(
            column.decimal_places)

  def format_floats(self, column_details: List[FormatFloatColumn]) -> None:
    """
        Apply formatting to float values in specified columns.

        Args:
            column_details (List[FormatFloatColumn]): A list of FormatFloatColumn instances specifying column and the format string.
        """
    for column_format in column_details:
      if column_format.column_name in self.data.columns:
        self.data[column_format.column_name] = self.data[
            column_format.column_name].apply(
                lambda x: column_format.format_string.format(x)
                if isinstance(x, float) else x)

  def format_strings(self, column_formats: List[FormatStringColumn]) -> None:
    """
        Apply string methods to specified columns.

        Args:
            column_formats (List[FormatStringColumn]): A list of FormatStringColumn instances specifying column and the string method to apply.
        """
    for column_format in column_formats:
      if column_format.column_name in self.data.columns:
        self.data[column_format.column_name] = self.data[
            column_format.column_name].apply(lambda x: getattr(
                x, column_format.string_method)() if isinstance(x, str) else x)

  def rename_columns(self, rename_details: List[RenameColumn]) -> None:
    """
        Rename specified columns.

        Args:
            rename_details (List[RenameColumn]): A list of RenameColumn instances specifying current and new column names.
        """
    rename_dict = {
        rename.current_name: rename.new_name
        for rename in rename_details
    }
    self.data.rename(columns=rename_dict, inplace=True)

  def sort_data(self, sort_details: List[SortDetails], axis: int = 0) -> None:
    """
        Sort the DataFrame by specified columns.

        Args:
            sort_details (List[SortDetails]): A list of SortDetails instances specifying column names and their corresponding sort order.
            axis (int): Axis to be sorted. 0 for index, 1 for columns.
        """
    sort_columns = [detail.column_name for detail in sort_details]
    sort_orders = [detail.ascending for detail in sort_details]
    self.data.sort_values(by=sort_columns,
                          axis=axis,
                          ascending=sort_orders,
                          inplace=True)

  def to_datetime(self, datetime_columns: List[ConvertToDateTime]) -> None:
    """
        Convert specified columns to datetime.

        Args:
            datetime_columns (List[ConvertToDateTime]): A list of ConvertToDateTime instances specifying column and the date format.
        """
    for column in datetime_columns:
      if column.column_name in self.data.columns:
        self.data[column.column_name] = pd.to_datetime(
            self.data[column.column_name], format=column.date_format)

  def format_datetime(self,
                      datetime_formats: List[FormatDateTimeColumn]) -> None:
    """
        Format datetime columns based on specified formats.

        Args:
            datetime_formats (List[FormatDateTimeColumn]): A list of FormatDateTimeColumn instances specifying column and the format string for datetime.
        """
    for datetime_format in datetime_formats:
      if datetime_format.column_name in self.data.columns:
        self.data[datetime_format.column_name] = self.data[
            datetime_format.column_name].dt.strftime(
                datetime_format.format_string)


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
  formatter = PandasDataFrameFormatter(df)

  # Example: Setting display options
  formatter.set_display_options(max_rows=10)

  # Example: Converting data types
  formatter.convert_type([
      ConvertColumnToType(column_name="A", dtype_name="float"),
  ])

  # Apply different rounding to different columns
  formatter.round_value([
      RoundColumnDecimalPlaces(column_name="B", decimal_places=3),
  ])

  # Apply different float formatting to different columns
  formatter.format_floats([
      FormatFloatColumn(column_name='A', format_string="{:.1f}"),
      FormatFloatColumn(column_name='B', format_string="{:.2f}"),
  ])

  # Apply different string methods to different columns
  formatter.format_strings([
      FormatStringColumn(column_name='C', string_method="upper"),
      FormatStringColumn(column_name='D', string_method="title"),
  ])

  # Rename multiple columns
  formatter.rename_columns([
      RenameColumn(current_name='A', new_name='Alpha'),
      RenameColumn(current_name='B', new_name='Beta'),
  ])

  # Convert string to datetime
  formatter.to_datetime([
      ConvertToDateTime(column_name='Date', date_format='%Y-%m-%d'),
  ])

  # Format multiple datetime columns with different formats
  formatter.format_datetime([
      FormatDateTimeColumn(column_name='Date',
                           format_string='%I:%M %p on %Y-%m-%d')
  ])

  formatter.sort_data([
      SortDetails(column_name='Beta', ascending=False),
      SortDetails(column_name='Alpha', ascending=True),
  ])

  print(formatter.data)
  print()
  print(formatter.data.dtypes)
