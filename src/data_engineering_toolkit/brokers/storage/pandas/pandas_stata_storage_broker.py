from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasStataStorageBroker(IStorageBroker):
    def create(
        self,
        dataframe: pd.DataFrame,
        filepath_or_buf: Union[str, bytes],
        *args,
        **kwargs
    ) -> None:
        """
        Write a DataFrame to a Stata file.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to Stata.
            filepath_or_buf (Union[str, bytes]): File path or object for the Stata file.
            *args: Additional positional arguments passed to pandas.DataFrame.to_stata.
            **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_stata.
        """
        dataframe.to_stata(filepath_or_buf, *args, **kwargs)

    def read(
        self, filepath_or_buffer: Union[str, bytes], *args, **kwargs
    ) -> pd.DataFrame:
        """
        Read a Stata file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object for the Stata file.
            *args: Additional positional arguments passed to pandas.read_stata.
            **kwargs: Additional keyword arguments passed to pandas.read_stata.

        Returns:
            pd.DataFrame: DataFrame read from the Stata file.
        """
        return pd.read_stata(filepath_or_buffer, *args, **kwargs)

    def update(
        self,
        filepath_or_buf: Union[str, bytes],
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Conceptually update records in a Stata file.

        This method reads the file, applies an update function to the DataFrame, and writes it back.

        Args:
            filepath_or_buf (Union[str, bytes]): File path or object for the Stata file.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            *args: Additional positional arguments for reading and writing.
            **kwargs: Additional keyword arguments for reading and writing.
        """
        # Read existing data
        df = pd.read_stata(filepath_or_buf, *args, **kwargs)

        # Apply the update function
        updated_df = update_function(df)

        # Write the updated DataFrame back to the file
        updated_df.to_stata(filepath_or_buf, *args, **kwargs)

    def delete(
        self,
        filepath_or_buf: Union[str, bytes],
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Conceptually delete records from a Stata file.

        This method reads the file, applies a delete function to the DataFrame, and writes it back.

        Args:
            filepath_or_buf (Union[str, bytes]): File path or object for the Stata file.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a DataFrame with records deleted.
            *args: Additional positional arguments for reading and writing.
            **kwargs: Additional keyword arguments for reading and writing.
        """
        # Read existing data
        df = pd.read_stata(filepath_or_buf, *args, **kwargs)

        # Apply the delete function
        updated_df = delete_function(df)

        # Write the updated DataFrame back to the file
        updated_df.to_stata(filepath_or_buf, *args, **kwargs)
