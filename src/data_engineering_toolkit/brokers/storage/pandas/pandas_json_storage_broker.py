from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasJSONStorageBroker(IStorageBroker):
    def create(
        self, dataframe: pd.DataFrame, path_or_buf: Union[str, bytes], *args, **kwargs
    ) -> None:
        """
        Write a DataFrame to a JSON file.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to JSON.
            path_or_buf (Union[str, bytes]): File path or object. If not specified, the result is returned as a string.
            *args: Additional positional arguments passed to pandas.to_json.
            **kwargs: Additional keyword arguments passed to pandas.to_json.
        """
        dataframe.to_json(path_or_buf=path_or_buf, *args, **kwargs)

    def read(
        self, filepath_or_buffer: Union[str, bytes], *args, **kwargs
    ) -> pd.DataFrame:
        """
        Read a JSON file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            *args: Additional positional arguments passed to pandas.read_json.
            **kwargs: Additional keyword arguments passed to pandas.read_json.

        Returns:
            pd.DataFrame: DataFrame read from the JSON file.
        """
        return pd.read_json(filepath_or_buffer, *args, **kwargs)

    def update(
        self,
        filepath_or_buffer: Union[str, bytes],
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Update records in a JSON file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            *args: Additional positional arguments passed to pandas.read_json and pandas.DataFrame.to_json.
            **kwargs: Additional keyword arguments passed to pandas.read_json and pandas.DataFrame.to_json.
        """
        # Read existing data
        df = pd.read_json(filepath_or_buffer, *args, **kwargs)

        # Update the dataframe using the provided function
        updated_df = update_function(df)

        # Write the updated dataframe back to the file
        updated_df.to_json(filepath_or_buffer, *args, **kwargs)

    def delete(
        self,
        filepath_or_buffer: Union[str, bytes],
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Delete records from a JSON file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a modified DataFrame with records deleted.
            *args: Additional positional arguments passed to pandas.read_json and pandas.DataFrame.to_json.
            **kwargs: Additional keyword arguments passed to pandas.read_json and pandas.DataFrame.to_json.
        """
        # Read existing data
        df = pd.read_json(filepath_or_buffer, *args, **kwargs)

        # Delete records from the dataframe using the provided function
        updated_df = delete_function(df)

        # Write the updated dataframe back to the file
        updated_df.to_json(filepath_or_buffer, *args, **kwargs)
