from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Callable

import pandas as pd

from models.formatting.pandas.pandas_formatter_models import (
    ConvertColumnToType,
    RoundColumnDecimalPlaces,
    FormatFloatColumn,
    FormatStringColumn,
    RenameColumn,
    ConvertToDateTime,
    FormatDateTimeColumn,
    SortDetails,
)
from services.util.enforce_attributes_meta import EnforceAttributesMeta

@dataclass
class IFormatter(metaclass=EnforceAttributesMeta):
    """Abstract base class for formatting dataframes in various ways."""

    __required_attributes__: List[str] = ['data']

    @abstractmethod
    def set_display_options(self, max_rows: Optional[int] = None, max_columns: Optional[int] = None, precision: Optional[int] = None) -> None:
        """Abstract method to set display options for the DataFrame."""
        pass

    @abstractmethod
    def apply_style(self, styler_function: Callable) -> pd.io.formats.style.Styler:
        """Abstract method to apply a styling function to the DataFrame."""
        pass

    @abstractmethod
    def convert_type(self, column_details: List[ConvertColumnToType]) -> None:
        """Abstract method to convert the data types of specified columns."""
        pass

    @abstractmethod
    def round_value(self, column_details: List[RoundColumnDecimalPlaces]) -> None:
        """Abstract method to round the values in specified columns to a set number of decimal places."""
        pass

    @abstractmethod
    def format_floats(self, column_details: List[FormatFloatColumn]) -> None:
        """Abstract method to apply formatting to float values in specified columns."""
        pass

    @abstractmethod
    def format_strings(self, column_formats: List[FormatStringColumn]) -> None:
        """Abstract method to apply string methods to specified columns."""
        pass

    @abstractmethod
    def rename_columns(self, rename_details: List[RenameColumn]) -> None:
        """Abstract method to rename specified columns."""
        pass

    @abstractmethod
    def sort_data(self, sort_details: List[SortDetails], axis: int = 0) -> None:
        """Abstract method to sort the DataFrame by specified columns."""
        pass

    @abstractmethod
    def to_datetime(self, datetime_columns: List[ConvertToDateTime]) -> None:
        """Abstract method to convert specified columns to datetime."""
        pass

    @abstractmethod
    def format_datetime(self, datetime_formats: List[FormatDateTimeColumn]) -> None:
        """Abstract method to format datetime columns based on specified formats."""
        pass
