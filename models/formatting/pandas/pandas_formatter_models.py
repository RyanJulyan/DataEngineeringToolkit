from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class ConvertColumnToType:
    column_name: str
    dtype_name: str


@dataclass
class RoundColumnDecimalPlaces:
    column_name: str
    decimal_places: int


@dataclass
class FormatFloatColumn:
    column_name: str
    format_string: str


@dataclass
class FormatStringColumn:
    column_name: str
    string_method: str


@dataclass
class RenameColumn:
    current_name: str
    new_name: str


@dataclass
class ConvertToDateTime:
    column_name: str
    date_format: Optional[str] = None


@dataclass
class FormatDateTimeColumn:
    column_name: str
    format_string: str


@dataclass
class SortDetails:
    column_name: str
    ascending: bool = True
