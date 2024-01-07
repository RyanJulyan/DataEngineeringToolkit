from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd
from google.cloud import bigquery

from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasGBQStorageBroker(IStorageBroker):
    project_id: str  # Google Cloud project ID
    client: bigquery.Client  # BigQuery client

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = bigquery.Client(project=self.project_id)

    def create(
        self, dataframe: pd.DataFrame, destination_table: str, *args, **kwargs
    ) -> None:
        """
        Write a DataFrame to a Google BigQuery table.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to BigQuery.
            destination_table (str): Full destination table name in the format `dataset.tablename`.
            *args: Additional positional arguments passed to pandas.to_gbq.
            **kwargs: Additional keyword arguments passed to pandas.to_gbq.
        """
        dataframe.to_gbq(destination_table, project_id=self.project_id, *args, **kwargs)

    def read(self, query: str, *args, **kwargs) -> pd.DataFrame:
        """
        Read data from Google BigQuery into a DataFrame.

        Args:
            query (str): SQL query to execute in BigQuery.
            *args: Additional positional arguments passed to pandas.read_gbq.
            **kwargs: Additional keyword arguments passed to pandas.read_gbq.

        Returns:
            pd.DataFrame: DataFrame containing the results of the query.
        """
        return pd.read_gbq(query, project_id=self.project_id, *args, **kwargs)

    # Update and delete operations in BigQuery are usually done via SQL queries.
    # Below are the conceptual methods for these, which are essentially wrappers around executing SQL queries.

    def update(self, query: str, *args, **kwargs) -> None:
        """
        Execute an update SQL query in Google BigQuery.

        Args:
            query (str): SQL update query to execute.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        job = self.client.query(query)
        job.result()  # Wait for the query to finish

    def delete(self, query: str, *args, **kwargs) -> None:
        """
        Execute a delete SQL query in Google BigQuery.

        Args:
            query (str): SQL delete query to execute.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        job = self.client.query(query)
        job.result()  # Wait for the query to finish
