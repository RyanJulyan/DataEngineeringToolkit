from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasExcelStorageBroker(IStorageBroker):
    def create(
        self, dataframe: pd.DataFrame, path_or_buf: Union[str, bytes], *args, **kwargs
    ) -> None:
        """
        Write a DataFrame to an Excel file.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to Excel.
            path_or_buf (Union[str, bytes]): File path or object. If not specified, the result is returned as a string.
            *args: Additional positional arguments passed to pandas.to_excel.
            **kwargs: Additional keyword arguments passed to pandas.to_excel.
        """
        dataframe.to_excel(path_or_buf=path_or_buf, *args, **kwargs)

    def read(
        self, filepath_or_buffer: Union[str, bytes], *args, **kwargs
    ) -> pd.DataFrame:
        """
        Read an Excel file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            *args: Additional positional arguments passed to pandas.read_excel.
            **kwargs: Additional keyword arguments passed to pandas.read_excel.

        Returns:
            pd.DataFrame: DataFrame read from the Excel file.
        """
        return pd.read_excel(filepath_or_buffer, *args, **kwargs)

    def update(
        self,
        filepath_or_buffer: Union[str, bytes],
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        sheet_name="Sheet1",
        *args,
        **kwargs
    ) -> None:
        """
        Update records in an Excel file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            sheet_name (str): Name of the Excel sheet to be updated.
            *args: Additional positional arguments passed to pandas.read_excel and pandas.DataFrame.to_excel.
            **kwargs: Additional keyword arguments passed to pandas.read_excel and pandas.DataFrame.to_excel.
        """
        # Read existing data
        df = pd.read_excel(filepath_or_buffer, sheet_name=sheet_name, *args, **kwargs)

        # Update the dataframe using the provided function
        updated_df = update_function(df)

        # Write the updated dataframe back to the file
        updated_df.to_excel(filepath_or_buffer, sheet_name=sheet_name, index=False)

    def delete(
        self,
        filepath_or_buffer: Union[str, bytes],
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        sheet_name="Sheet1",
        *args,
        **kwargs
    ) -> None:
        """
        Delete records from an Excel file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a modified DataFrame with records deleted.
            sheet_name (str): Name of the Excel sheet where records will be deleted.
            *args: Additional positional arguments passed to pandas.read_excel and pandas.DataFrame.to_excel.
            **kwargs: Additional keyword arguments passed to pandas.read_excel and pandas.DataFrame.to_excel.
        """
        # Read existing data
        df = pd.read_excel(filepath_or_buffer, sheet_name=sheet_name, *args, **kwargs)

        # Delete records from the dataframe using the provided function
        updated_df = delete_function(df)

        # Write the updated dataframe back to the file
        updated_df.to_excel(filepath_or_buffer, sheet_name=sheet_name, index=False)
