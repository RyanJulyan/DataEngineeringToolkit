from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import pandas as pd

# Models
from data_engineering_toolkit.models.transformations.pandas.pandas_transformations_models import (
    AddRowOperation,
    Aggregation,
    ApplyFunctionOperation,
    ArithmeticOperation,
    ConcatOperation,
    ConditionalApplyOperation,
    ConditionalOperation,
    CrosstabOperation,
    DropRowOperation,
    GroupByOperation,
    HandleMissingDataColumn,
    MapValuesOperation,
    MergeOperation,
    ModifyRowOperation,
    NewColumn,
    PivotTableOperation,
    ResampleOperation,
    RollingWindowOperation,
    SelectConditionOperation,
    StringContainsOperation,
)


@dataclass
class PandasDataFrameTransformations:
    """
    A class for performing various transformations on a pandas DataFrames in different ways.
    """

    data: pd.DataFrame

    def select_columns(self, columns: List[str]) -> PandasDataFrameTransformations:
        """
        Limit the columns of the DataFrame.

        Args:
            columns (List[str]): A list of column names to return and modify DataFrame.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place.
        """
        self.data = self.data[columns]
        return self

    # TODO: Review if this is really what I want as it breaks the pattern!
    def select_rows_by_index(self, index: int):
        """
        Select rows based on specified index number.

        Args:
            index (int): The index to search and return data based on
        """
        return self.data.loc[index:index]

    def select_rows_by_conditions(
        self, condition_operations: List[SelectConditionOperation]
    ) -> PandasDataFrameTransformations:
        """
        Select rows based on specified conditions and operations.

        This method iterates over a list of SelectConditionOperation instances,
        each specifying a condition and an operation to be applied on the DataFrame.

        NOTE: SelectConditionOperation.condition must all start with the data as the first input value,
              SelectConditionOperation.kwargs are passed after the data as named vars.

        Args:
            condition_operations (List[SelectConditionOperation]): A list of SelectConditionOperation instances,
                                                             each specifying a condition and an operation.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place.
        """
        for cond_op in condition_operations:
            self.data = self.data[cond_op.condition(self.data, **cond_op.kwargs)]

        return self

    def handle_missing_values(
        self, column_methods: List[HandleMissingDataColumn]
    ) -> PandasDataFrameTransformations:
        """
        Handle missing data in specified DataFrame columns using specified methods.

        This method iterates over a list of HandleMissingDataColumn instances,
        each specifying a column and how to handle its missing data. The handling
        can be either filling missing data (fillna) or dropping rows with missing data (dropna).

        Args:
            column_methods (List[HandleMissingDataColumn]): A list of HandleMissingDataColumn instances,
                                                           each specifying a column and method for handling missing data.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place.
        """
        for column_method in column_methods:
            if column_method.column_name in self.data.columns:
                if column_method.method == "fillna":
                    self.data[column_method.column_name] = self.data[
                        column_method.column_name
                    ].fillna(column_method.value, **column_method.kwargs)
                elif column_method.method == "dropna":
                    self.data.dropna(
                        subset=[column_method.column_name],
                        inplace=True,
                        **column_method.kwargs,
                    )

        return self

    def group_by(self, columns: List[str]) -> PandasDataFrameTransformations:
        """
        Group the DataFrame by the specified columns.

        This method groups the DataFrame based on the values in the specified columns.
        It's useful for performing group-wise analysis and aggregations.

        Args:
            columns (List[str]): A list of column names to group the DataFrame by.

        Returns:
            PandasDataFrameTransformations: An instance of the class with the DataFrame grouped by the specified columns.
            Note that the actual grouping operation is lazy and doesn't compute anything until an aggregation is applied.
        """

        self.data.groupby(columns)

        return self

    def sort_by_index(self) -> PandasDataFrameTransformations:
        """
        Sort values in the DataFrame by the index

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place.
        """
        self.data = self.data.sort_index()

        return self

    def add_new_columns(
        self, new_columns: List[NewColumn]
    ) -> PandasDataFrameTransformations:
        """
        Add new columns to the DataFrame based on specified criteria.

        This method allows adding multiple new columns to the DataFrame. Each new column can be
        defined with either a list of values or a function that generates values based on the DataFrame.

        Args:
            new_columns (List[NewColumn]): A list of NewColumn instances, each specifying a new column name
                                           and values or a function to generate the values.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by adding new columns.
        """
        for new_column in new_columns:
            if callable(new_column.values):
                self.data[new_column.column_name] = new_column.values(self.data)
            else:
                self.data[new_column.column_name] = new_column.values

        return self

    def drop_columns(
        self, columns_to_drop: List[str]
    ) -> PandasDataFrameTransformations:
        """
        Drop specified columns from the DataFrame.

        This method allows for dropping multiple columns from the DataFrame based on the provided list of
        strings, each specifying a column name to be dropped.

        Args:
            columns_to_drop (List[str]): A list of column names to be dropped.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by dropping specified columns.
        """
        for column in columns_to_drop:
            self.data = self.data.drop(column, axis=1)

        return self

    def merge_dataframes(
        self, merge_operations: List[MergeOperation]
    ) -> PandasDataFrameTransformations:
        """
        Merge the current DataFrame with other DataFrames based on specified criteria.

        This method allows for merging the current DataFrame with multiple other DataFrames.
        Each merge operation is defined with the other DataFrame, the column to merge on,
        and the merge method (how).

        Args:
            merge_operations (List[MergeOperation]): A list of MergeOperation instances,
                                                     each specifying an other DataFrame,
                                                     a column name to merge on, and a merge method.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by merging with other DataFrames.
        """
        for operation in merge_operations:
            self.data = pd.merge(
                self.data,
                operation.other_df,
                on=operation.on_column,
                how=operation.how,
                **operation.kwargs,
            )

        return self

    def concat_dataframes(
        self, concat_operations: List[ConcatOperation]
    ) -> PandasDataFrameTransformations:
        """
        Concatenate the current DataFrame with other DataFrames based on specified criteria.

        This method allows for concatenating the current DataFrame with multiple other DataFrames.
        Each concat operation can specify the axis, join method, and whether to ignore the index.

        Args:
            concat_operations (List[ConcatOperation]): A list of ConcatOperation instances,
                                                       each specifying an other DataFrame and the concat parameters.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by concatenating it with other DataFrames.
        """
        for operation in concat_operations:
            self.data = pd.concat(
                [self.data, operation.other_df],
                axis=operation.axis,
                join=operation.join,
                ignore_index=operation.ignore_index,
                **operation.kwargs,
            )

        return self

    # TODO: Review if this is really what I want as it breaks the pattern!
    def create_pivot_tables(
        self, pivot_operations: List[PivotTableOperation]
    ) -> List[pd.DataFrame]:
        """
        Create pivot tables from the DataFrame based on specified criteria.

        This method allows for creating multiple pivot tables from the DataFrame.
        Each pivot operation can specify the values, index, columns, aggregation function,
        value to fill missing entries, and whether to include margins.

        Args:
            pivot_operations (List[PivotTableOperation]): A list of PivotTableOperation instances,
                                                          each specifying parameters for creating a pivot table.

        Returns:
            List[pd.DataFrame]: A list of pivot tables created as per the specified operations.
        """
        pivot_tables = []
        for operation in pivot_operations:
            pivot_table = self.data.pivot_table(
                values=operation.values,
                index=operation.index,
                columns=operation.columns,
                aggfunc=operation.aggfunc,
                fill_value=operation.fill_value,
                margins=operation.margins,
                **operation.kwargs,
            )
            pivot_tables.append(pivot_table)

        return pivot_tables

    # TODO: Review if this is really what I want as it breaks the pattern!
    def create_crosstabs(
        self, crosstab_operations: List[CrosstabOperation]
    ) -> List[pd.DataFrame]:
        """
        Create cross-tabulations from the DataFrame based on specified criteria.

        This method allows for creating multiple cross-tabulations from the DataFrame.
        Each crosstab operation can specify the columns for the cross-tabulation,
        optional aggregation values and function, and whether to include margins.

        Args:
            crosstab_operations (List[CrosstabOperation]): A list of CrosstabOperation instances,
                                                          each specifying parameters for creating a cross-tabulation.

        Returns:
            List[pd.DataFrame]: A list of cross-tabulations created as per the specified operations.
        """
        crosstabs = []
        for operation in crosstab_operations:
            crosstab = pd.crosstab(
                self.data[operation.col1],
                self.data[operation.col2],
                values=self.data[operation.values] if operation.values else None,
                aggfunc=operation.aggfunc,
                margins=operation.margins,
                **operation.kwargs,
            )
            crosstabs.append(crosstab)

        return crosstabs

    def apply_functions(
        self, apply_operations: List[ApplyFunctionOperation]
    ) -> PandasDataFrameTransformations:
        """
        Apply functions to specified columns in the DataFrame based on given criteria.

        Args:
            apply_operations (List[ApplyFunctionOperation]): A list of ApplyFunctionOperation instances,
                                                             each specifying a function, columns to apply it to, and additional parameters.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by applying the functions.
        """
        for operation in apply_operations:
            if operation.columns:
                if isinstance(
                    operation.columns, str
                ):  # If a single column is specified
                    self.data[operation.columns] = self.data[operation.columns].apply(
                        operation.func,
                        axis=operation.axis,
                        raw=operation.raw,
                        result_type=operation.result_type,
                        **operation.kwargs,
                    )
                else:  # If multiple columns are specified
                    self.data[operation.columns] = self.data[operation.columns].apply(
                        operation.func,
                        axis=operation.axis,
                        raw=operation.raw,
                        result_type=operation.result_type,
                        **operation.kwargs,
                    )
            else:
                self.data = self.data.apply(
                    operation.func,
                    axis=operation.axis,
                    raw=operation.raw,
                    result_type=operation.result_type,
                    **operation.kwargs,
                )

        return self

    def string_contains(
        self, string_operations: List[StringContainsOperation]
    ) -> PandasDataFrameTransformations:
        """
        Check if a substring is contained within a string in specified DataFrame columns.

        This method allows for checking the containment of a substring in multiple DataFrame columns.
        Each string operation can specify the column to check and the substring to look for.

        Args:
            string_operations (List[StringContainsOperation]): A list of StringContainsOperation instances,
                                                              each specifying a column and a substring.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by adding new columns with boolean values indicating the presence of the substring.
        """
        for operation in string_operations:
            new_column_name = operation.new_column_format.format(**operation.__dict__)
            self.data[new_column_name] = self.data[operation.column_name].str.contains(
                operation.substring,
                na=operation.na,
                **operation.kwargs,
            )

        return self

    def conditional_operations(
        self, conditional_operations: List[ConditionalOperation]
    ) -> PandasDataFrameTransformations:
        """
        Apply conditional operations based on specified criteria for DataFrame columns.

        This method allows for applying multiple conditional operations on DataFrame columns.
        Each operation can specify a column, a condition function, values to use if the condition is true or false,
        and an optional new column name to store the results.

        Args:
            conditional_operations (List[ConditionalOperation]): A list of ConditionalOperation instances,
                                                                each specifying a column, condition function, true value, false value,
                                                                and an optional new column name.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by applying the conditional operations.
        """
        for operation in conditional_operations:
            result_column = (
                operation.new_column if operation.new_column else operation.column
            )
            self.data[result_column] = np.where(
                operation.condition_func(self.data[operation.column]),
                operation.true_val,
                operation.false_val,
            )

        return self

    # Aggregation methods
    def group_by_and_transform(
        self,
        group_column: str,
        new_column: str,
        agg_func: Union[
            List[
                Literal[
                    "any",
                    "all",
                    "count",
                    "cov",
                    "first",
                    "idxmax",
                    "idxmin",
                    "last",
                    "max",
                    "mean",
                    "median",
                    "min",
                    "nunique",
                    "prod",
                    "quantile",
                    "sem",
                    "size",
                    "skew",
                    "std",
                    "sum",
                    "var",
                ]
            ],
            Callable[[Any, Any], Any],
        ],
    ) -> PandasDataFrameTransformations:
        """
        Applies a transformation function to the specified column, partitioning the data
        by the given group column. This is similar to PARTITION BY in SQL.

        Args:
          group_column (str): Column to group by.
          new_column (str): Name of the new column to be added with the result.
          agg_func (): Aggregation function to apply (e.g., 'mean', 'sum', custom function).
        """
        self.data[new_column] = self.data.groupby(group_column)[new_column].transform(
            agg_func
        )

        return self

    def group_by_and_aggregate(
        self, group_by_operations: List[GroupByOperation]
    ) -> PandasDataFrameTransformations:
        """
        Group by specified columns and apply aggregation functions.

        Args:
            group_by_operations (List[GroupByOperation]): A list of GroupByOperation instances.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place.
        """
        for operation in group_by_operations:
            aggregation_dict = {
                agg.column_name: agg.aggregation_function
                for agg in operation.aggregations
            }
            grouped = self.data.groupby(operation.group_by_columns)
            self.data = grouped.agg(aggregation_dict).reset_index()

        return self

    def aggregate_data(
        self, aggregate_operations: List[Aggregation]
    ) -> PandasDataFrameTransformations:
        """
        Aggregate the DataFrame based on specified aggregation functions.

        This method allows for applying multiple aggregation functions to the DataFrame.
        Each aggregation operation can specify the function and optionally the columns to aggregate.

        Args:
            aggregate_operations (List[Aggregation]): A list of Aggregation instances,
                                                             each specifying an aggregation function and optionally columns.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by applying the aggregation functions.
        """

        aggregation_dict = {
            agg.column_name: agg.aggregation_function for agg in aggregate_operations
        }
        self.data = self.data.aggregate(aggregation_dict)

        return self

    def transform_columns(
        self, transform_operations: List[Aggregation]
    ) -> PandasDataFrameTransformations:
        """
        Apply transformation functions to specified columns in the DataFrame.

        This method allows for applying transformation functions to multiple DataFrame columns.
        Each transform operation can specify the column and the transformation function to be applied.

        Args:
            transform_operations (List[TransformOperation]): A list of TransformOperation instances,
                                                             each specifying a column and a transformation function.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by applying the transformation functions.
        """
        for operation in transform_operations:
            self.data[operation.column_name] = self.data[
                operation.column_name
            ].transform(operation.aggregation_function)

        return self

    # Row operations
    def add_rows(
        self, add_row_operations: List[AddRowOperation]
    ) -> PandasDataFrameTransformations:
        """
        Add new rows to the DataFrame based on specified data.

        This method allows for adding multiple new rows to the DataFrame.
        Each add row operation specifies the data for the new row in a dictionary format,
        with column names as keys.

        Args:
            add_row_operations (List[AddRowOperation]): A list of AddRowOperation instances,
                                                       each specifying data for a new row.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by adding new rows.
        """
        for operation in add_row_operations:
            new_row = pd.DataFrame([operation.row_data], columns=self.data.columns)
            self.data = pd.concat([self.data, new_row], ignore_index=True)

        return self

    def modify_rows(
        self, modify_row_operations: List[ModifyRowOperation]
    ) -> PandasDataFrameTransformations:
        """
        Modify existing rows in the DataFrame based on specified conditions and data.

        This method allows for modifying multiple rows in the DataFrame.
        Each modify row operation specifies a condition function to identify rows and the new data for those rows.

        Args:
            modify_row_operations (List[ModifyRowOperation]): A list of ModifyRowOperation instances,
                                                             each specifying a condition to identify rows and new data for those rows.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by updating specified rows.
        """
        for operation in modify_row_operations:
            condition = operation.condition_func(self.data, **operation.kwargs)
            self.data.loc[condition, list(operation.row_data.keys())] = pd.DataFrame(
                [operation.row_data], index=self.data[condition].index
            )

        return self

    def drop_rows(
        self, drop_row_operations: List[DropRowOperation]
    ) -> PandasDataFrameTransformations:
        """
        Drop rows from the DataFrame based on specified conditions.

        This method allows for dropping multiple rows in the DataFrame.
        Each drop row operation specifies a condition function to identify which rows should be dropped.

        Args:
            drop_row_operations (List[DropRowOperation]): A list of DropRowOperation instances,
                                                         each specifying a condition to identify rows for deletion.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by dropping specified rows.
        """
        for operation in drop_row_operations:
            condition = operation.condition_func(self.data)
            self.data = self.data.drop(self.data[condition].index, **operation.kwargs)

        return self

    # Time series analysis
    def resample_data(
        self, resample_operations: ResampleOperation
    ) -> PandasDataFrameTransformations:
        """
        Resample the DataFrame based on specified rules and aggregation functions.

        This method allows for resampling the DataFrame using different rules and applying
        various aggregation functions. Each resample operation specifies the resampling rule
        and the aggregation function or functions to use.

        Args:
            resample_operations (ResampleOperation): A ResampleOperation instances,
                                                          specifying a resampling rule and aggregation function(s).

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by applying the resampling operations.
        """
        aggregation_dict = {
            agg.column_name: agg.aggregation_function
            for agg in resample_operations.aggregations
        }
        self.data = self.data.resample(resample_operations.rule).agg(aggregation_dict)

        return self

    def rolling_window_operations(
        self, rolling_window_operations: List[RollingWindowOperation]
    ) -> PandasDataFrameTransformations:
        """
        Apply rolling window operations to specified columns in the DataFrame with individual window sizes and aggregation functions.

        This method allows for applying rolling window operations with different sizes and aggregation functions to different DataFrame columns.
        Each operation specifies the column, window size, aggregation function, and new_column_format.

        Args:
            rolling_window_operations (List[RollingWindowOperation]): A list of RollingWindowOperation instances,
                                                                     each specifying a column, window size, aggregation function, and new_column_format.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by applying the rolling window operations.
        """
        for operation in rolling_window_operations:
            new_column_name = operation.new_column_format.format(**operation.__dict__)
            self.data[new_column_name] = (
                self.data[operation.column_name]
                .rolling(window=operation.window)
                .agg(operation.agg_func)
            )

        return self

    def arithmetic_operations(
        self, arithmetic_operations: List[ArithmeticOperation]
    ) -> PandasDataFrameTransformations:
        """
        Perform arithmetic operations on specified columns in the DataFrame.

        This method allows for applying multiple arithmetic operations to DataFrame columns.
        Each arithmetic operation specifies the column, the operation (add, subtract, multiply, divide),
        and the value to use in the operation.

        Args:
            arithmetic_operations (List[ArithmeticOperation]): A list of ArithmeticOperation instances,
                                                              each specifying a column, operation, and value.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by performing arithmetic operations.
        """
        for operation in arithmetic_operations:
            if operation.operation == "add":
                self.data[operation.column] += operation.value
            elif operation.operation == "subtract":
                self.data[operation.column] -= operation.value
            elif operation.operation == "multiply":
                self.data[operation.column] *= operation.value
            elif operation.operation == "divide":
                self.data[operation.column] /= operation.value

        return self

    # More advanced mapping
    def map_values(
        self, map_values_operations: List[MapValuesOperation]
    ) -> PandasDataFrameTransformations:
        """
        Map values of specified columns in the DataFrame based on given mapping dictionaries.

        This method allows for applying mapping operations to multiple DataFrame columns.
        Each mapping operation specifies the column and the dictionary used for mapping values in that column.

        Args:
            map_values_operations (List[MapValuesOperation]): A list of MapValuesOperation instances,
                                                             each specifying a column and a mapping dictionary.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by applying the mapping operations.
        """
        for operation in map_values_operations:
            self.data[operation.column] = self.data[operation.column].map(
                arg=operation.mapping_dict,
                na_action=operation.na_action,
            )

        return self

    # More complex conditional operations
    def conditional_apply(
        self, conditional_apply_operations: List[ConditionalApplyOperation]
    ) -> PandasDataFrameTransformations:
        """
        Apply functions conditionally to specified columns in the DataFrame.

        This method allows for conditionally applying functions to multiple DataFrame columns.
        Each conditional apply operation specifies the column, a condition function to identify rows,
        and a function to apply to those rows.

        Args:
            conditional_apply_operations (List[ConditionalApplyOperation]): A list of ConditionalApplyOperation instances,
                                                                           each specifying a column, condition function, and apply function.

        Returns:
            PandasDataFrameTransformations: The method modifies the DataFrame in place by conditionally applying the functions.
        """
        for operation in conditional_apply_operations:
            condition = operation.condition_func(self.data, **operation.kwargs)
            self.data.loc[condition, operation.column] = self.data.loc[
                condition, operation.column
            ].apply(operation.apply_func)

        return self

    def feature_engineering(
        self, new_feature_name: str, function: Callable, *args: Any
    ) -> PandasDataFrameTransformations:
        """
        Creates a new feature in the DataFrame based on a specified function.

        Args:
            new_feature_name (str): The name of the new feature to be created.
            function (Callable): The function to apply to each row to generate the new feature.
            *args (Any): Additional arguments to pass to the function.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[new_feature_name] = self.dataframe.apply(
            lambda row: function(row, *args), axis=1
        )

        return self


if __name__ == "__main__":
    # Example Usage
    import operator
    import numpy as np
    from data_engineering_toolkit.models.transformations.pandas.pandas_transformations_models import (
        ColumnOperationFact,
    )

    data = {
        "A": [1.23, 2.34, 3.45, 1.23],
        "B": [4.567672, 5, 6.78, 4.567672],
        "C": ["foo", "bar", "baz", "foo"],
        "D": ["dog", "elephant", "frog", "dog"],
        "E": [1, np.nan, 3, 1],
        "F": [4, 5, np.nan, 4],
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-01"],
    }
    df = pd.DataFrame(data)
    transformations = PandasDataFrameTransformations(df)

    print("select_rows_by_index")
    print(transformations.select_rows_by_index(2))
    print()

    # Define another DataFrame to concatenate
    other_data = {
        "A": [5.67, 7.89],
        "B": [1.234, 3.456],
        "C": ["xyz", "abc"],
        "D": ["cat", "mouse"],
        "E": [7, 8],
        "F": [9, 10],
        "Date": ["2023-01-04", "2023-01-05"],
    }
    other_df = pd.DataFrame(other_data)

    # Define concat operation
    concat_op = ConcatOperation(
        other_df=other_df, axis=0, join="outer", ignore_index=True
    )

    # Apply the concat_dataframes method
    transformations.concat_dataframes([concat_op])

    # Print the resulting DataFrame
    print("concat_dataframes")
    print(transformations.data)
    print()

    # Apply different missing data handling methods to different columns
    transformations.handle_missing_values(
        [
            HandleMissingDataColumn(column_name="E", method="fillna", value=0),
            HandleMissingDataColumn(column_name="F", method="dropna"),
        ]
    )

    print("handle_missing_values")
    print(transformations.data)
    print()

    # Define a function to generate values for a new column
    def calculate_new_column(df):
        return df["A"] * 2  # Example operation

    # Define new columns to add
    new_columns = [
        NewColumn(
            column_name="NewColumn1", values=[1, 2, 3, 4, 5]
        ),  # Adding a column with a list of values
        NewColumn(
            column_name="NewColumn2", values=calculate_new_column
        ),  # Adding a column using a function
    ]

    # Apply the add_new_columns method
    transformations.add_new_columns(new_columns)
    print("add_new_columns")
    print(transformations.data)
    print()

    # Define row data to add
    add_row_op = AddRowOperation(
        row_data={
            "A": 5.67,
            "B": 1.234,
            "C": "xyz",
            "D": "cat",
            "E": 7,
            "F": 9,
            "Date": "2023-01-24",
        }
    )

    # Apply the add_rows method
    transformations.add_rows([add_row_op])

    # Print the resulting DataFrame
    print("add_rows")
    print(transformations.data)
    print()

    # Define arithmetic operation
    arithmetic_op = ArithmeticOperation(
        column="A",  # Column to apply the arithmetic operation
        operation="add",  # Operation type
        value=10,  # Value for the operation
    )

    # Apply the arithmetic_operations method
    transformations.arithmetic_operations([arithmetic_op])

    # Print the resulting DataFrame
    print("arithmetic_operations")
    print(transformations.data)
    print()

    # Define pivot table operation
    pivot_op = PivotTableOperation(
        values=["A", "B"],
        index=["C"],
        columns=["D"],
        aggfunc="sum",
        fill_value=0,
        margins=True,
    )

    # Create pivot tables
    pivot_tables = transformations.create_pivot_tables([pivot_op])

    # Print the resulting pivot tables
    print("create_pivot_tables")
    for table in pivot_tables:
        print(table)
    print()

    # Define crosstab operation
    crosstab_op = CrosstabOperation(
        col1="A", col2="B", values=["C"], aggfunc="sum", margins=True
    )

    # Create crosstabs
    crosstabs = transformations.create_crosstabs([crosstab_op])

    # Print the resulting crosstabs
    print("create_crosstabs")
    for table in crosstabs:
        print(table)
    print()

    # Apply the drop_columns method
    transformations.drop_columns(columns_to_drop=["NewColumn1"])

    # Print the resulting data
    print("drop_columns")
    print(transformations.data)
    print()

    # Define string contains operation
    string_contains_op = StringContainsOperation(column_name="C", substring="foo")

    # Apply the string_contains method
    transformations.string_contains([string_contains_op])

    # Print the resulting DataFrame
    print("string_contains")
    print(transformations.data)
    print()

    # Create a SelectConditionOperation instance using generic_dataframe_column_operation_fact function
    condition_operation = SelectConditionOperation(
        kwargs=ColumnOperationFact(column_name="E", op=operator.ge, fact=1).__dict__,
    )

    # Apply the select_rows_by_conditions method
    selected_rows = transformations.select_rows_by_conditions([condition_operation])

    # Display the selected rows
    print("select_rows_by_conditions")
    print(transformations.data)
    print()

    # Define a condition
    def condition_greater_than_one(df):
        return df["E"] >= 1

    # Create a ConditionOperation instance
    condition_operation = SelectConditionOperation(
        condition=condition_greater_than_one,
    )

    # Apply the select_rows_by_conditions method
    selected_rows = transformations.select_rows_by_conditions([condition_operation])

    # Display the selected rows
    print("select_rows_by_conditions")
    print(transformations.data)
    print()

    # Define a function to apply
    def example_function(x):
        return x * 10  # Example function logic

    # Define apply function operation for specific columns
    apply_op = ApplyFunctionOperation(
        func=example_function,
        columns=["A", "B"],  # Apply function only to columns 'A' and 'B'
        axis=0,
    )

    # Apply the function to the DataFrame
    transformations.apply_functions([apply_op])

    # Display the selected rows
    print("apply_functions")
    print(transformations.data)
    print()

    # Define a condition function
    def condition_for_apply(df):
        return df["A"] > 120  # Example condition

    # Define a function to apply
    def apply_function(x):
        return x * 2  # Example function logic

    # Define conditional apply operation
    conditional_apply_op = ConditionalApplyOperation(
        column="B",  # Column to apply the function
        condition_func=condition_for_apply,  # Condition to identify rows
        apply_func=apply_function,  # Function to apply
    )

    # Apply the conditional_apply method
    transformations.conditional_apply([conditional_apply_op])

    # Print the resulting DataFrame
    print("conditional_apply")
    print(transformations.data)
    print()

    # Define a condition function
    def greater_than_condition(x, threshold=2):
        return x > threshold

    # Define conditional operation
    conditional_op = ConditionalOperation(
        column="A",
        condition_func=lambda x: greater_than_condition(
            x, 25
        ),  # Using a lambda for the condition
        true_val="Greater",
        false_val="Lesser",
        new_column="A_GreaterThan2",  # Optional new column to store the results
    )

    # Apply the conditional_operations method
    transformations.conditional_operations([conditional_op])

    # Print the resulting DataFrame
    print("conditional_operations")
    print(transformations.data)
    print()

    # Define a transformation function
    def example_transformation(x):
        return x * 2  # Example transformation logic

    # Define transform operation
    transform_op = Aggregation(
        column_name="A",  # Column to apply the transformation
        aggregation_function=example_transformation,  # Transformation function
    )

    # Apply the transform_columns method
    transformations.transform_columns(transform_operations=[transform_op])

    # Print the resulting DataFrame
    print("transform_columns")
    print(transformations.data)
    print()

    # Define mapping dictionary
    mapping_dict = {
        "foo": "new_foo",
        "bar": "new_bar",
        "baz": "new_baz",
    }  # Example mapping

    # Define map values operation
    map_values_op = MapValuesOperation(
        column="C", mapping_dict=mapping_dict  # Column to apply the mapping
    )

    # Apply the map_values method
    transformations.map_values([map_values_op])

    # Print the resulting DataFrame
    print("map_values")
    print(transformations.data)
    print()

    # Define a condition function to identify rows
    def condition_to_modify(df):
        return df["C"] == "new_foo"  # Example condition

    # Define row data to modify
    modify_row_op = ModifyRowOperation(
        condition_func=condition_to_modify,  # Condition to identify rows
        row_data={
            "A": 8.90,
            "B": 2.345,
            "D": "new_animal",
            "E": 10,
        },  # Data for modifying the row
    )

    # Apply the modify_rows method
    transformations.modify_rows([modify_row_op])

    # Print the resulting DataFrame
    print("modify_rows")
    print(transformations.data)
    print()

    # Define a condition function to identify rows for deletion
    def condition_to_drop(df):
        return df["C"] == "foo"  # Example condition

    # Define drop row operation
    drop_row_op = DropRowOperation(
        condition_func=condition_to_drop  # Condition to identify rows for deletion
    )

    # Apply the drop_rows method
    transformations.drop_rows([drop_row_op])

    # Print the resulting DataFrame
    print("drop_rows")
    print(transformations.data)
    print()

    # Apply different missing data handling methods to different columns
    transformations.select_columns(
        [
            "A",
            "B",
            "C",
            "D",
        ]
    )
    print("select_columns")
    print(transformations.data)
    print()

    transformations.group_by(["A", "D"])

    print("group_by")
    print(transformations.data)
    print()

    transformations2 = PandasDataFrameTransformations(transformations.data)

    # Define aggregations
    aggregations = [
        Aggregation(column_name="B", aggregation_function="sum"),
        Aggregation(column_name="C", aggregation_function="count"),
    ]

    # Apply the aggregate_data method
    transformations.aggregate_data(aggregations)

    # Print the resulting DataFrame
    print("aggregate_data")
    print(transformations.data)
    print()

    # Define a GroupByOperation
    group_by_op = GroupByOperation(
        group_by_columns=["A", "D"], aggregations=aggregations
    )

    # Apply the group_by_and_aggregate method
    transformations2.group_by_and_aggregate([group_by_op])
    print("group_by_and_aggregate")
    print(transformations2.data)
    print()

    # Create a date range
    date_range = pd.date_range(start="2023-01-01", end="2023-01-31", freq="H")

    # Generate sample data
    np.random.seed(0)  # For reproducible results
    data = {
        "Temperature": np.random.uniform(
            20, 30, len(date_range)
        ),  # Random temperatures between 20 and 30
        "Humidity": np.random.uniform(
            30, 60, len(date_range)
        ),  # Random humidity values between 30 and 60
        "WindSpeed": np.random.uniform(
            5, 20, len(date_range)
        ),  # Random wind speeds between 5 and 20
    }

    # Create DataFrame
    df = pd.DataFrame(data, index=date_range)

    # Example usage of PandasDataFrameTransformations class
    transformations = PandasDataFrameTransformations(df)

    print("Original Timeseries DataFrame:")
    print(df)
    print()

    # Define aggregations
    aggregations = [
        Aggregation(column_name="Temperature", aggregation_function="sum"),
        Aggregation(column_name="Humidity", aggregation_function="count"),
        Aggregation(column_name="WindSpeed", aggregation_function="mean"),
    ]

    # Define resample operation
    resample_op = ResampleOperation(
        rule="D",
        aggregations=aggregations,  # Resampling daily  # Calculating different aggregation_functions per column
    )

    # Apply the resample_data method
    transformations.resample_data(resample_op)

    # Print the resulting DataFrame
    print("resample_data")
    print(transformations.data)
    print()

    # Define rolling window operations for individual columns
    rolling_window_ops = [
        RollingWindowOperation(column_name="Temperature", window=3, agg_func="mean"),
        RollingWindowOperation(column_name="Humidity", window=5, agg_func="sum"),
    ]

    # Apply the rolling_window_operations method
    transformations.rolling_window_operations(rolling_window_ops)

    # Print the resulting DataFrame
    print("rolling_window_operations")
    print(transformations.data)
    print()
