import numpy as np
import pandas as pd

# Models

from data_engineering_toolkit.models.formatting.pandas.pandas_formatter_models import (
    ConvertColumnToType,
    ConvertToDateTime,
    FormatDateTimeColumn,
    FormatFloatColumn,
    FormatStringColumn,
    RenameColumn,
    RoundColumnDecimalPlaces,
    SortDetails,
)

# Services
from data_engineering_toolkit.services.formatting.pandas.pandas_formatter import (
    PandasDataFrameFormatter, )
from data_engineering_toolkit.services.transformations.pandas.pandas_transformations import (
    PandasDataFrameTransformations, )

data = {
    "A": [1.23, 2.34, 3.45],
    "B": [4.567672, 5, 6.78],
    "C": ["foo", "bar", "baz"],
    "D": ["dog", "elephant", "frog"],
    "E": [1, np.nan, 3],
    "F": [4, 5, np.nan],
    "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
}
df = pd.DataFrame(data)
formatter = PandasDataFrameFormatter(df)

# Example: Setting display options
formatter.set_display_options(max_rows=10)

# Example: Converting data types
formatter.convert_data_type([
    ConvertColumnToType(column_name="A", dtype_name="float"),
])

# Apply different rounding to different columns
formatter.round_value([
    RoundColumnDecimalPlaces(column_name="B", decimal_places=3),
])

# Apply different float formatting to different columns
formatter.format_floats([
    FormatFloatColumn(column_name="A", format_string="{:.1f}"),
    FormatFloatColumn(column_name="B", format_string="{:.2f}"),
])

# Apply different string methods to different columns
formatter.format_strings([
    FormatStringColumn(column_name="C", string_method="upper"),
    FormatStringColumn(column_name="D", string_method="title"),
])

# Rename multiple columns
formatter.rename_columns([
    RenameColumn(current_name="A", new_name="Alpha"),
    RenameColumn(current_name="B", new_name="Beta"),
])

# Convert string to datetime
formatter.to_datetime([
    ConvertToDateTime(column_name="Date", date_format="%Y-%m-%d"),
])

# Format multiple datetime columns with different formats
formatter.format_datetime([
    FormatDateTimeColumn(column_name="Date",
                         format_string="%I:%M %p on %Y-%m-%d")
])

formatter.sort_data([
    SortDetails(column_name="Beta", ascending=False),
    SortDetails(column_name="Alpha", ascending=True),
])

print(formatter.data)
print()
print(formatter.data.dtypes)
print()

df = formatter.data
transformations = PandasDataFrameTransformations(df)

print(transformations.data)
print()
print(transformations.data.dtypes)
