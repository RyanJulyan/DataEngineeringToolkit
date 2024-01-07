from dataclasses import dataclass
from typing import Any, Callable, Dict, Union

import pandas as pd

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasFWFStorageBroker(IStorageBroker):
    def create(
        self,
        dataframe: pd.DataFrame,
        path_or_buf: Union[str, bytes],
        column_widths: Dict[str, int],
        *args,
        **kwargs,
    ) -> None:
        """
        Write a DataFrame to a fixed-width formatted (FWF) file.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to FWF.
            path_or_buf (Union[str, bytes]): File path or object. If not specified, the result is returned as a string.
            column_widths (Dict[str, int]): The DataFrame to format. e.g. {"Name": 20, "Age": 3, "City": 15}
            *args: Additional positional arguments passed to pandas.to_fwf (not available in pandas, need to use `to_csv` with formatting).
            **kwargs: Additional keyword arguments passed to pandas.to_fwf (not available in pandas, need to use `to_csv` with formatting).
        """
        # Note: Pandas does not have a direct to_fwf method. We may need to format the DataFrame appropriately and then use to_csv.
        formatted_df = self.format_dataframe_for_fwf(
            dataframe=dataframe, column_widths=column_widths
        )
        formatted_df.to_csv(path_or_buf=path_or_buf, sep="\t", *args, **kwargs)

    def read(
        self,
        filepath_or_buffer: Union[str, bytes],
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Read a fixed-width formatted (FWF) file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            *args: Additional positional arguments passed to pandas.read_fwf.
            **kwargs: Additional keyword arguments passed to pandas.read_fwf.

        Returns:
            pd.DataFrame: DataFrame read from the FWF file.
        """
        return pd.read_fwf(filepath_or_buffer, *args, **kwargs)

    def update(
        self,
        filepath_or_buffer: Union[str, bytes],
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        column_widths: Dict[str, int],
        *args,
        **kwargs,
    ) -> None:
        """
        Update records in a fixed-width formatted (FWF) file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            column_widths (Dict[str, int]): The DataFrame to format. e.g. {"Name": 20, "Age": 3, "City": 15}
            *args: Additional positional arguments passed to pandas.read_fwf.
            **kwargs: Additional keyword arguments passed to pandas.read_fwf.
        """
        # Read existing data
        df = pd.read_fwf(filepath_or_buffer, *args, **kwargs)

        # Update the dataframe using the provided function
        updated_df = update_function(df)

        # Write the updated dataframe back to the file
        formatted_df = self.format_dataframe_for_fwf(
            dataframe=updated_df, column_widths=column_widths
        )
        formatted_df.to_csv(filepath_or_buffer, sep="\t", index=False)

    def delete(
        self,
        filepath_or_buffer: Union[str, bytes],
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        column_widths: Dict[str, int],
        *args,
        **kwargs,
    ) -> None:
        """
        Delete records from a fixed-width formatted (FWF) file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a modified DataFrame with records deleted.
            column_widths (Dict[str, int]): The DataFrame to format. e.g. {"Name": 20, "Age": 3, "City": 15}
            *args: Additional positional arguments passed to pandas.read_fwf.
            **kwargs: Additional keyword arguments passed to pandas.read_fwf.
        """
        # Read existing data
        df = pd.read_fwf(filepath_or_buffer, *args, **kwargs)

        # Delete records from the dataframe using the provided function
        updated_df = delete_function(df)

        # Write the updated dataframe back to the file
        formatted_df = self.format_dataframe_for_fwf(
            dataframe=updated_df, column_widths=column_widths
        )
        formatted_df.to_csv(filepath_or_buffer, sep="\t", index=False)

    def format_dataframe_for_fwf(
        self,
        dataframe: pd.DataFrame,
        column_widths: Dict[str, int],
    ) -> pd.DataFrame:
        """
        Formats a DataFrame to fixed-width format. This is a helper method to prepare the DataFrame for writing as FWF.

        Args:
            dataframe (pd.DataFrame): The DataFrame to format.
            column_widths (Dict[str, int]): The DataFrame to format. e.g. {"Name": 20, "Age": 3, "City": 15}

        Returns:
            pd.DataFrame: The formatted DataFrame.
        """
        # Format each column
        for column, width in column_widths.items():
            # Ensure the column is of string type
            dataframe[column] = dataframe[column].astype(str)

            # Truncate the data if it's longer than the specified width
            dataframe[column] = dataframe[column].str.slice(0, width)

            # Pad the data if it's shorter than the specified width
            dataframe[column] = dataframe[column].str.pad(width, side="right")

        # Concatenate the columns into a single string column
        formatted_df = dataframe.apply(lambda row: "".join(row), axis=1)

        return formatted_df
