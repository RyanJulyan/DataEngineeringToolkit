from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasHDF5StorageBroker(IStorageBroker):
    def create(
        self,
        dataframe: pd.DataFrame,
        path_or_buf: Union[str, bytes],
        key: str,
        *args,
        **kwargs
    ) -> None:
        """
        Write a DataFrame to an HDF5 file.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to HDF5.
            path_or_buf (Union[str, bytes]): File path or object.
            key (str): Identifier for the group in the store.
            *args: Additional positional arguments passed to pandas.DataFrame.to_hdf.
            **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_hdf.
        """
        dataframe.to_hdf(path_or_buf, key, *args, **kwargs)

    def read(
        self, filepath_or_buffer: Union[str, bytes], key: str, *args, **kwargs
    ) -> pd.DataFrame:
        """
        Read an HDF5 file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            key (str): Identifier for the group in the store.
            *args: Additional positional arguments passed to pandas.read_hdf.
            **kwargs: Additional keyword arguments passed to pandas.read_hdf.

        Returns:
            pd.DataFrame: DataFrame read from the HDF5 file.
        """
        return pd.read_hdf(filepath_or_buffer, key, *args, **kwargs)

    def update(
        self,
        filepath_or_buffer: Union[str, bytes],
        key: str,
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Update records in an HDF5 file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            key (str): Identifier for the group in the store.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Read existing data
        df = pd.read_hdf(filepath_or_buffer, key, *args, **kwargs)

        # Update the DataFrame using the provided function
        updated_df = update_function(df)

        # Overwrite the data in the HDF5 file with the updated DataFrame
        updated_df.to_hdf(filepath_or_buffer, key, mode="w", *args, **kwargs)

    def delete(
        self,
        filepath_or_buffer: Union[str, bytes],
        key: str,
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Delete records from an HDF5 file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            key (str): Identifier for the group in the store.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a modified DataFrame with records deleted.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Read existing data
        df = pd.read_hdf(filepath_or_buffer, key, *args, **kwargs)

        # Delete records from the DataFrame using the provided function
        updated_df = delete_function(df)

        # Overwrite the data in the HDF5 file with the updated DataFrame
        updated_df.to_hdf(filepath_or_buffer, key, mode="w", *args, **kwargs)
