from dataclasses import dataclass
from typing import Any, Callable, Union
import xml.etree.ElementTree as ET

import pandas as pd

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasXMLStorageBroker(IStorageBroker):
    def create(
        self,
        dataframe: pd.DataFrame,
        path_or_buf: Union[str, bytes],
        root_tag: str,
        row_tag: str,
        *args,
        **kwargs
    ) -> None:
        """
        Write a DataFrame to an XML file.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to XML.
            path_or_buf (Union[str, bytes]): File path or object.
            root_tag (str): The root tag for the XML document.
            row_tag (str): The tag for each row in the DataFrame.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        root = ET.Element(root_tag)
        for _, row in dataframe.iterrows():
            row_element = ET.SubElement(root, row_tag)
            for col in dataframe.columns:
                col_element = ET.SubElement(row_element, col)
                col_element.text = str(row[col])

        tree = ET.ElementTree(root)
        tree.write(path_or_buf, *args, **kwargs)

    def read(
        self,
        filepath_or_buffer: Union[str, bytes],
        root_tag: str,
        row_tag: str,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read an XML file into a DataFrame.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            root_tag (str): The root tag for the XML document.
            row_tag (str): The tag for each row in the XML.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: DataFrame read from the XML file.
        """
        tree = ET.parse(filepath_or_buffer)
        root = tree.getroot()

        # Assuming that the XML structure is flat and simple
        data = []
        for elem in root.findall(row_tag):
            row_data = {}
            for child in elem:
                row_data[child.tag] = child.text
            data.append(row_data)

        return pd.DataFrame(data)

    def update(
        self,
        filepath_or_buffer: Union[str, bytes],
        update_function: Callable[[pd.DataFrame], pd.DataFrame],
        root_tag: str,
        row_tag: str,
        *args,
        **kwargs
    ) -> None:
        """
        Update records in an XML file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            update_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns an updated DataFrame.
            root_tag (str): The root tag for the XML document.
            row_tag (str): The tag for each row in the XML.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Read the existing XML file into a DataFrame
        df = self.read(filepath_or_buffer, root_tag, row_tag, *args, **kwargs)

        # Update the DataFrame using the provided function
        updated_df = update_function(df)

        # Write the updated DataFrame back to the XML file
        self.create(updated_df, filepath_or_buffer, root_tag, row_tag, *args, **kwargs)

    def delete(
        self,
        filepath_or_buffer: Union[str, bytes],
        delete_function: Callable[[pd.DataFrame], pd.DataFrame],
        root_tag: str,
        row_tag: str,
        *args,
        **kwargs
    ) -> None:
        """
        Delete records from an XML file.

        Args:
            filepath_or_buffer (Union[str, bytes]): File path or object.
            delete_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame and returns a modified DataFrame with records deleted.
            root_tag (str): The root tag for the XML document.
            row_tag (str): The tag for each row in the XML.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Read the existing XML file into a DataFrame
        df = self.read(filepath_or_buffer, root_tag, row_tag, *args, **kwargs)

        # Delete records from the DataFrame using the provided function
        updated_df = delete_function(df)

        # Write the updated DataFrame back to the XML file
        self.create(updated_df, filepath_or_buffer, root_tag, row_tag, *args, **kwargs)
