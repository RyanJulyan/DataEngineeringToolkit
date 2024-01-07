from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Callable

# Services
from data_engineering_toolkit.services.util.enforce_attributes_meta import (
    EnforceAttributesMeta,
)


@dataclass
class ITransformations(metaclass=EnforceAttributesMeta):
    """
    Abstract base class for transforming dataframes in various ways.
    This class defines a set of abstract methods that must be implemented
    by any concrete subclass to perform various operations on pandas DataFrames.
    """

    __required_attributes__ = ["data"]

    @abstractmethod
    def select_columns(self, columns: List[str]) -> Any:
        """
        Select specific columns from the DataFrame.

        Args:
            columns (List[str]): A list of column names to be selected.

        Returns:
            Any: The transformed DataFrame with only the selected columns.
        """
        pass

    @abstractmethod
    def select_rows_by_index(self, index: int) -> Any:
        """
        Select rows based on a specific index.

        Args:
            index (int): The index number to select rows.

        Returns:
            Any: The transformed DataFrame with rows selected by index.
        """
        pass

    @abstractmethod
    def select_rows_by_conditions(self, condition_operations: List[Any]) -> Any:
        """
        Select rows from the DataFrame based on specified conditions.

        Args:
            condition_operations (List[Any]): A list of condition operations to apply for row selection.

        Returns:
            Any: The transformed DataFrame after applying the row selection conditions.
        """
        pass

    @abstractmethod
    def filter_data_boolean(self, condition: Callable) -> Any:
        """
        Filter data using a boolean condition.

        Args:
            condition (Callable): A function that takes the DataFrame and returns a boolean series for filtering.

        Returns:
            Any: The transformed DataFrame after applying the boolean filter.
        """
        pass

    @abstractmethod
    def filter_data_query(self, query_string: str) -> Any:
        """
        Filter data using a query string.

        Args:
            query_string (str): The query string to filter data.

        Returns:
            Any: The DataFrame after applying the query-based filter.
        """
        pass

    @abstractmethod
    def join_dataframes(self, other_df: Any, on: str, how: str) -> Any:
        """
        Join the current DataFrame with another DataFrame.

        Args:
            other_df (Any): The DataFrame to join with.
            on (str): The column name to join on.
            how (str): Type of join - 'left', 'right', 'inner', 'outer'.

        Returns:
            Any: The DataFrame after performing the join.
        """
        pass

    @abstractmethod
    def group_by(self, columns: List[str]) -> Any:
        """
        Group the DataFrame by specified columns.

        Args:
            columns (List[str]): Column names to group by.

        Returns:
            Any: The DataFrame grouped by the specified columns.
        """
        pass

    @abstractmethod
    def add_new_columns(self, new_columns: List[Any]) -> Any:
        """
        Add new columns to the DataFrame.

        Args:
            new_columns (List[Any]): Specifications for new columns to add.

        Returns:
            Any: The DataFrame with the new columns added.
        """
        pass

    @abstractmethod
    def drop_columns(self, columns_to_drop: List[str]) -> Any:
        """
        Drop specified columns from the DataFrame.

        Args:
            columns_to_drop (List[str]): Column names to drop.

        Returns:
            Any: The DataFrame with specified columns dropped.
        """
        pass

    @abstractmethod
    def merge_dataframes(self, merge_operations: List[Any]) -> Any:
        """
        Merge the current DataFrame with other DataFrames.

        Args:
            merge_operations (List[Any]): Specifications for merging with other DataFrames.

        Returns:
            Any: The DataFrame after performing the merge operations.
        """
        pass

    @abstractmethod
    def concat_dataframes(self, concat_operations: List[Any]) -> Any:
        """
        Concatenate the current DataFrame with other DataFrames.

        Args:
            concat_operations (List[Any]): Specifications for concatenation.

        Returns:
            Any: The DataFrame after performing the concatenation.
        """
        pass

    @abstractmethod
    def create_pivot_tables(self, pivot_operations: List[Any]) -> List[Any]:
        """
        Create pivot tables from the DataFrame.

        Args:
            pivot_operations (List[Any]): Specifications for creating pivot tables.

        Returns:
            List[Any]: A list of pivot tables created as per the specified operations.
        """
        pass

    @abstractmethod
    def create_crosstabs(self, crosstab_operations: List[Any]) -> List[Any]:
        """
        Create cross-tabulations from the DataFrame.

        Args:
            crosstab_operations (List[Any]): Specifications for creating cross-tabulations.

        Returns:
            List[Any]: A list of cross-tabulations created as per the specified operations.
        """
        pass

    @abstractmethod
    def apply_functions(self, apply_operations: List[Any]) -> Any:
        """
        Apply specified functions to the DataFrame.

        Args:
            apply_operations (List[Any]): A list of operations, each defining a function to apply to the DataFrame.

        Returns:
            Any: The DataFrame after applying the specified functions.
        """
        pass

    @abstractmethod
    def string_contains(self, string_operations: List[Any]) -> Any:
        """
        Check if specified substrings are contained within DataFrame columns.

        Args:
            string_operations (List[Any]): A list of operations for substring search.

        Returns:
            Any: The DataFrame with results of substring search operations.
        """
        pass

    @abstractmethod
    def conditional_operations(self, conditional_operations: List[Any]) -> Any:
        """
        Apply conditional operations to the DataFrame.

        Args:
            conditional_operations (List[Any]): A list of conditional operations to apply.

        Returns:
            Any: The DataFrame after applying the conditional operations.
        """
        pass

    @abstractmethod
    def group_by_and_transform(
        self, group_column: str, new_column: str, agg_func: Any
    ) -> Any:
        """
        Group the DataFrame by a column and apply a transformation function.

        Args:
            group_column (str): The column to group by.
            new_column (str): The name of the new column to be added.
            agg_func (Any): The aggregation function to apply.

        Returns:
            Any: The DataFrame after applying the group-by and transformation.
        """
        pass

    @abstractmethod
    def group_by_and_aggregate(self, group_by_operations: List[Any]) -> Any:
        """
        Group the DataFrame by specified columns and apply aggregation functions.

        Args:
            group_by_operations (List[Any]): A list of group-by operations with aggregation.

        Returns:
            Any: The DataFrame after applying the group-by and aggregation operations.
        """
        pass

    @abstractmethod
    def aggregate_data(self, aggregate_operations: List[Any]) -> Any:
        """
        Aggregate the DataFrame based on specified functions.

        Args:
            aggregate_operations (List[Any]): A list of aggregation operations.

        Returns:
            Any: The DataFrame after applying the aggregation operations.
        """
        pass

    @abstractmethod
    def transform_columns(self, transform_operations: List[Any]) -> Any:
        """
        Apply transformation functions to specified columns in the DataFrame.

        Args:
            transform_operations (List[Any]): A list of column transformation operations.

        Returns:
            Any: The DataFrame after applying the transformations to the columns.
        """
        pass

    @abstractmethod
    def add_rows(self, add_row_operations: List[Any]) -> Any:
        """
        Add new rows to the DataFrame.

        Args:
            add_row_operations (List[Any]): A list of operations defining new rows to add.

        Returns:
            Any: The DataFrame after adding new rows.
        """
        pass

    @abstractmethod
    def modify_rows(self, modify_row_operations: List[Any]) -> Any:
        """
        Modify existing rows in the DataFrame based on specified conditions.

        Args:
            modify_row_operations (List[Any]): A list of operations for row modification.

        Returns:
            Any: The DataFrame after modifying the rows.
        """
        pass

    @abstractmethod
    def drop_rows(self, drop_row_operations: List[Any]) -> Any:
        """
        Drop rows from the DataFrame based on specified conditions.

        Args:
            drop_row_operations (List[Any]): A list of operations defining conditions for row deletion.

        Returns:
            Any: The DataFrame after dropping the specified rows.
        """
        pass

    @abstractmethod
    def resample_data(self, resample_operations: Any) -> Any:
        """
        Resample the DataFrame based on specified rules and aggregation functions.

        Args:
            resample_operations (Any): Resampling operations with rules and aggregation functions.

        Returns:
            Any: The DataFrame after applying the resampling operations.
        """
        pass

    @abstractmethod
    def rolling_window_operations(self, rolling_window_operations: List[Any]) -> Any:
        """
        Apply rolling window operations to specified columns.

        Args:
            rolling_window_operations (List[Any]): A list of rolling window operations.

        Returns:
            Any: The DataFrame after applying the rolling window operations.
        """
        pass

    @abstractmethod
    def arithmetic_operations(self, arithmetic_operations: List[Any]) -> Any:
        """
        Perform arithmetic operations on specified DataFrame columns.

        Args:
            arithmetic_operations (List[Any]): A list of arithmetic operations.

        Returns:
            Any: The DataFrame after performing the arithmetic operations.
        """
        pass

    @abstractmethod
    def map_values(self, map_values_operations: List[Any]) -> Any:
        """
        Map values of specified columns based on given dictionaries.

        Args:
            map_values_operations (List[Any]): A list of mapping operations.

        Returns:
            Any: The DataFrame after applying the mapping operations.
        """
        pass

    @abstractmethod
    def conditional_apply(self, conditional_apply_operations: List[Any]) -> Any:
        """
        Apply functions conditionally to specified DataFrame columns.

        Args:
            conditional_apply_operations (List[Any]): A list of conditional apply operations.

        Returns:
            Any: The DataFrame after conditionally applying functions to columns.
        """
        pass

    @abstractmethod
    def feature_engineering(
        self, new_feature_name: str, function: Callable, *args: Any
    ) -> Any:
        """
        Create a new feature in the DataFrame based on a specified function.

        Args:
            new_feature_name (str): The name of the new feature.
            function (Callable): Function to generate the new feature.
            *args (Any): Additional arguments for the function.

        Returns:
            Any: The DataFrame with the new feature added.
        """
        pass
