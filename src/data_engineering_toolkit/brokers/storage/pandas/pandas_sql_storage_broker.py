from dataclasses import dataclass
from typing import Any, Callable, Union

import pandas as pd
from sqlalchemy import create_engine

from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker


@dataclass
class PandasSQLStorageBroker(IStorageBroker):
    engine: Any  # SQLAlchemy engine

    def create(self, dataframe: pd.DataFrame, table_name: str, *args, **kwargs) -> None:
        """
        Write a DataFrame to a SQL table.

        Args:
            dataframe (pd.DataFrame): DataFrame to be written to SQL.
            table_name (str): Name of the SQL table.
            *args: Additional positional arguments passed to pandas.DataFrame.to_sql.
            **kwargs: Additional keyword arguments passed to pandas.DataFrame.to_sql.
        """
        dataframe.to_sql(table_name, self.engine, *args, **kwargs)

    def read(self, sql_query: str, *args, **kwargs) -> pd.DataFrame:
        """
        Read from a SQL table or execute a SQL query into a DataFrame.

        Args:
            sql_query (str): SQL query or table name to execute.
            *args: Additional positional arguments passed to pandas.read_sql.
            **kwargs: Additional keyword arguments passed to pandas.read_sql.

        Returns:
            pd.DataFrame: DataFrame resulting from the SQL query.
        """
        return pd.read_sql(sql_query, self.engine, *args, **kwargs)

    def update(self, sql_query: str, *args, **kwargs) -> None:
        """
        Execute an update SQL query.

        Args:
            sql_query (str): SQL update query to execute.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        with self.engine.connect() as connection:
            connection.execute(sql_query, *args, **kwargs)

    def delete(self, sql_query: str, *args, **kwargs) -> None:
        """
        Execute a delete SQL query.

        Args:
            sql_query (str): SQL delete query to execute.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        with self.engine.connect() as connection:
            connection.execute(sql_query, *args, **kwargs)
