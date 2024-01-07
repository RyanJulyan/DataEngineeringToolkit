from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd

from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasSASStorageBroker(IStorageBroker):
    def create(
        self, dataframe: pd.DataFrame, path_or_buf: Union[str, bytes], *args, **kwargs
    ) -> None:
        """
        Write a DataFrame to a SAS file.

        Note: Writing to SAS files is not natively supported by Pandas.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to SAS.
            path_or_buf (Union[str, bytes]): File path or object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "Writing to SAS files is not natively supported by Pandas."
        )

    def read(
        self, filepath_or_buffer: Union[str, bytes], *args, **kwargs
    ) -> pd.DataFrame:
        """
        Read a SAS file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            *args: Additional positional arguments passed to pandas.read_sas.
            **kwargs: Additional keyword arguments passed to pandas.read_sas.

        Returns:
            pd.DataFrame: DataFrame read from the SAS file.
        """
        return pd.read_sas(filepath_or_buffer, *args, **kwargs)

    def update(
        self,
        filepath_or_buffer: Union[str, bytes],
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Update records in a SAS file. (Hypothetical implementation)

        Note: This method is not typically used and is provided for conceptual purposes, as writing to SAS files is not supported by Pandas.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "Writing to SAS files is not natively supported by Pandas."
        )

    def delete(
        self,
        filepath_or_buffer: Union[str, bytes],
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        *args,
        **kwargs
    ) -> None:
        """
        Delete records from a SAS file. (Hypothetical implementation)

        Note: This method is not typically used and is provided for conceptual purposes, as writing to SAS files is not supported by Pandas.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a modified DataFrame with records deleted.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "Writing to SAS files is not natively supported by Pandas."
        )
