from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasORCStorageBroker(IStorageBroker):
    def create(
        self, dataframe: pd.DataFrame, path_or_buf: Union[str, bytes], *args, **kwargs
    ) -> None:
        """
        Write a DataFrame to an ORC file.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to ORC.
            path_or_buf (Union[str, bytes]): File path or object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        dataframe.to_orc(path_or_buf, *args, **kwargs)

    def read(
        self, filepath_or_buffer: Union[str, bytes], *args, **kwargs
    ) -> pd.DataFrame:
        """
        Read an ORC file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: DataFrame read from the ORC file.
        """
        return pd.read_orc(filepath_or_buffer, *args, **kwargs)

    def update(
        self,
        filepath_or_buffer: Union[str, bytes],
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Update records in an ORC file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Read existing data
        df = pd.read_orc(filepath_or_buffer, *args, **kwargs)

        # Update the DataFrame using the provided function
        updated_df = update_function(df)

        # Write the updated DataFrame back to the ORC file
        updated_df.to_orc(filepath_or_buffer, *args, **kwargs)

    def delete(
        self,
        filepath_or_buffer: Union[str, bytes],
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Delete records from an ORC file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a modified DataFrame with records deleted.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Read existing data
        df = pd.read_orc(filepath_or_buffer, *args, **kwargs)

        # Delete records from the DataFrame using the provided function
        updated_df = delete_function(df)

        # Write the updated DataFrame back to the ORC file
        updated_df.to_orc(filepath_or_buffer, *args, **kwargs)
