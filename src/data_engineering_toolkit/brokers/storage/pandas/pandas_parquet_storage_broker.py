from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasParquetStorageBroker(IStorageBroker):
    def create(
        self, dataframe: pd.DataFrame, path_or_buf: Union[str, bytes], *args, **kwargs
    ) -> None:
        """
        Write a DataFrame to a Parquet file.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to Parquet.
            path_or_buf (Union[str, bytes]): File path or object.
            *args: Additional positional arguments passed to pandas.DataFrame.to_parquet.
            **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_parquet.
        """
        dataframe.to_parquet(path_or_buf, *args, **kwargs)

    def read(
        self, filepath_or_buffer: Union[str, bytes], *args, **kwargs
    ) -> pd.DataFrame:
        """
        Read a Parquet file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            *args: Additional positional arguments passed to pandas.read_parquet.
            **kwargs: Additional keyword arguments passed to pandas.read_parquet.

        Returns:
            pd.DataFrame: DataFrame read from the Parquet file.
        """
        return pd.read_parquet(filepath_or_buffer, *args, **kwargs)

    def update(
        self,
        filepath_or_buffer: Union[str, bytes],
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Update records in a Parquet file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Read existing data
        df = pd.read_parquet(filepath_or_buffer, *args, **kwargs)

        # Update the DataFrame using the provided function
        updated_df = update_function(df)

        # Write the updated DataFrame back to the Parquet file
        updated_df.to_parquet(filepath_or_buffer, *args, **kwargs)

    def delete(
        self,
        filepath_or_buffer: Union[str, bytes],
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Delete records from a Parquet file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a modified DataFrame with records deleted.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Read existing data
        df = pd.read_parquet(filepath_or_buffer, *args, **kwargs)

        # Delete records from the DataFrame using the provided function
        updated_df = delete_function(df)

        # Write the updated DataFrame back to the Parquet file
        updated_df.to_parquet(filepath_or_buffer, *args, **kwargs)
