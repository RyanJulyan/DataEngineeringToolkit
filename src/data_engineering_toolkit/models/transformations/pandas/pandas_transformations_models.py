from dataclasses import dataclass, field
import operator
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import pandas as pd


def generic_dataframe_column_operation_fact(
    df: pd.DataFrame, op: operator, column_name: str, fact: Any
):
    return op(df[column_name], fact)


@dataclass
class AddRowOperation:
    row_data: Dict[
        str, Any
    ]  # Data for the new row as a dictionary, with column names as keys


@dataclass
class Aggregation:
    column_name: str
    aggregation_function: Union[
        List[
            Literal[
                "any",  # Compute whether any of the values in the groups are truthy
                "all",  # Compute whether all of the values in the groups are truthy
                "count",  # Compute the number of non-NA values in the groups
                "cov",  # Compute the covariance of the groups
                "first",  # Compute the first occurring value in each group
                "idxmax",  # Compute the index of the maximum value in each group
                "idxmin",  # Compute the index of the minimum value in each group
                "last",  # Compute the last occurring value in each group
                "max",  # Compute the maximum value in each group
                "mean",  # Compute the mean of each group
                "median",  # Compute the median of each group
                "min",  # Compute the minimum value in each group
                "nunique",  # Compute the number of unique values in each group
                "prod",  # Compute the product of the values in each group
                "quantile",  # Compute a given quantile of the values in each group
                "sem",  # Compute the standard error of the mean of the values in each group
                "size",  # Compute the number of values in each group
                "skew",  # Compute the skew of the values in each group
                "std",  # Compute the standard deviation of the values in each group
                "sum",  # Compute the sum of the values in each group
                "var",  # Compute the variance of the values in each group
            ]
        ],
        Callable[[Any, Any], Any],
    ]


@dataclass
class ApplyFunctionOperation:
    func: Callable
    columns: Optional[
        Union[str, List[str]]
    ] = None  # Columns to apply the function to, None applies to all columns
    axis: int = 0  # Apply function to each column (0) or row (1)
    raw: bool = (
        False  # Raw data passed to function as a Series if False, as a ndarray if True
    )
    result_type: Optional[
        Literal["reduce", "broadcast", "expand"]
    ] = None  # Choose result type, e.g., 'reduce', 'broadcast', 'expand'
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class ArithmeticOperation:
    column: str
    operation: Literal["add", "subtract", "multiply", "divide"]
    value: Any  # The value to be used in the arithmetic operation


@dataclass
class ColumnOperationFact:
    column_name: str
    op: operator
    fact: Any


@dataclass
class ConditionalOperation:
    column: str
    condition_func: Callable[
        [Any], Any
    ]  # Function that takes a value and returns a boolean
    true_val: Any
    false_val: Any
    new_column: Optional[
        str
    ] = None  # Optional name for the new column to store results


@dataclass
class ConditionalApplyOperation:
    column: str
    condition_func: Callable[[pd.DataFrame], pd.Series]  # Function to identify rows
    apply_func: Callable[[Any], Any]  # Function to apply to the identified rows
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class CrosstabOperation:
    col1: str
    col2: str
    values: Optional[List[str]] = None  # Optional values for aggregation
    aggfunc: Optional[Union[str, Callable]] = None  # Optional aggregation function
    margins: bool = False  # Add all row/columns (e.g., for subtotals)
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class ConcatOperation:
    other_df: pd.DataFrame
    axis: int = 0  # Default axis (0 for rows, 1 for columns)
    join: Literal["inner", "outer"] = "outer"  # Default join method
    ignore_index: bool = False  # Default behavior for index
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class DropRowOperation:
    condition_func: Callable[
        [pd.DataFrame], pd.Series
    ]  # Function to identify rows for deletion
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class GroupByOperation:
    group_by_columns: List[str]
    aggregations: List[Aggregation]


@dataclass
class MapValuesOperation:
    column: str
    mapping_dict: Dict[
        Any, Any
    ]  # Dictionary for mapping values in the specified column
    na_action: Literal["ignore"] | None = None


@dataclass
class MergeOperation:
    other_df: pd.DataFrame
    on_column: str
    how: Literal["inner", "outer"] = "inner"  # Default merge method
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class ModifyRowOperation:
    condition_func: Callable[[pd.DataFrame], pd.Series]  # Function to identify rows
    row_data: Dict[str, Any]  # Data for modifying the row, with column names as keys
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class NewColumn:
    column_name: str
    values: Union[List[Any], Callable[[pd.DataFrame], pd.Series]]


@dataclass
class PivotTableOperation:
    values: List[str]
    index: List[str]
    columns: List[str]
    aggfunc: Optional[
        Union[str, Callable, List]
    ] = "mean"  # Default aggregation function
    fill_value: Optional[Any] = None  # Value to replace missing values
    margins: bool = False  # Add all row/columns (e.g., for subtotals)
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class ResampleOperation:
    rule: str  # The offset string or object representing target conversion
    aggregations: List[Aggregation]


@dataclass
class RollingWindowOperation:
    column_name: str
    window: int  # The number of periods for the rolling window
    agg_func: Union[
        List[
            Literal[
                "any",  # Compute whether any of the values in the groups are truthy
                "all",  # Compute whether all of the values in the groups are truthy
                "count",  # Compute the number of non-NA values in the groups
                "cov",  # Compute the covariance of the groups
                "first",  # Compute the first occurring value in each group
                "idxmax",  # Compute the index of the maximum value in each group
                "idxmin",  # Compute the index of the minimum value in each group
                "last",  # Compute the last occurring value in each group
                "max",  # Compute the maximum value in each group
                "mean",  # Compute the mean of each group
                "median",  # Compute the median of each group
                "min",  # Compute the minimum value in each group
                "nunique",  # Compute the number of unique values in each group
                "prod",  # Compute the product of the values in each group
                "quantile",  # Compute a given quantile of the values in each group
                "sem",  # Compute the standard error of the mean of the values in each group
                "size",  # Compute the number of values in each group
                "skew",  # Compute the skew of the values in each group
                "std",  # Compute the standard deviation of the values in each group
                "sum",  # Compute the sum of the values in each group
                "var",  # Compute the variance of the values in each group
            ]
        ],
        Callable[[Any, Any], Any],
    ]  # Aggregation function or functions
    new_column_format: str = "{column_name}_rolling_window"


@dataclass
class SelectConditionOperation:
    condition: Callable[
        [pd.DataFrame], pd.Series
    ] = generic_dataframe_column_operation_fact
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class StringContainsOperation:
    column_name: str
    substring: str
    new_column_format: str = "{column_name}_contains_{substring}"
    na: bool = False  # Boolean to handle NaN values; if False NaNs are treated as False
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function
