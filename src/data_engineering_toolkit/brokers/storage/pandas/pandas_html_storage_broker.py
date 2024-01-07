from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasHTMLStorageBroker(IStorageBroker):
    def create(
        self, dataframe: pd.DataFrame, path_or_buf: Union[str, bytes], *args, **kwargs
    ) -> None:
        """
        Write a DataFrame to an HTML file.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to HTML.
            path_or_buf (Union[str, bytes]): File path or object. If not specified, the result is returned as a string.
            *args: Additional positional arguments passed to pandas.to_html.
            **kwargs: Additional keyword arguments passed to pandas.to_html.
        """
        dataframe.to_html(path_or_buf=path_or_buf, *args, **kwargs)

    def read(
        self, filepath_or_buffer: Union[str, bytes], *args, **kwargs
    ) -> pd.DataFrame:
        """
        Read an HTML file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            *args: Additional positional arguments passed to pandas.read_html.
            **kwargs: Additional keyword arguments passed to pandas.read_html.

        Returns:
            pd.DataFrame: DataFrame read from the HTML file.
        """
        # Note: read_html returns a list of DataFrames. Assuming the first table is required.
        return pd.read_html(filepath_or_buffer, *args, **kwargs)[0]

    def update(
        self,
        filepath_or_buffer: Union[str, bytes],
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Update records in an HTML file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            *args: Additional positional arguments passed to pandas.read_html and pandas.DataFrame.to_html.
            **kwargs: Additional keyword arguments passed to pandas.read_html and pandas.DataFrame.to_html.
        """
        # Read existing data
        df_list = pd.read_html(filepath_or_buffer, *args, **kwargs)
        if df_list:
            df = df_list[0]  # Assuming the first table is the one to update

            # Update the dataframe using the provided function
            updated_df = update_function(df)

            # Write the updated dataframe back to the file
            updated_df.to_html(filepath_or_buffer, *args, **kwargs)

    def delete(
        self,
        filepath_or_buffer: Union[str, bytes],
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Delete records from an HTML file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a modified DataFrame with records deleted.
            *args: Additional positional arguments passed to pandas.read_html and pandas.DataFrame.to_html.
            **kwargs: Additional keyword arguments passed to pandas.read_html and pandas.DataFrame.to_html.
        """
        # Read existing data
        df_list = pd.read_html(filepath_or_buffer, *args, **kwargs)
        if df_list:
            df = df_list[0]  # Assuming the first table is the one to delete from

            # Delete records from the dataframe using the provided function
            updated_df = delete_function(df)

            # Write the updated dataframe back to the file
            updated_df.to_html(filepath_or_buffer, *args, **kwargs)
