from dataclasses import dataclass, field
import operator
from typing import Any, Callable, Dict, Optional

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import pandas as pd


def generic_dataframe_column_operation_fact(
    op: operator, df: pd.DataFrame, column_name: str, fact: Any
):
    return op(df[column_name], fact)


@dataclass
class ColumnOperationFact:
    column_name: str
    op: operator
    fact: Any


@dataclass
class ConditionOperation:
    condition: Callable[
        [pd.DataFrame], pd.Series
    ] = generic_dataframe_column_operation_fact
    operation_type: Literal["filter"] = "filter"
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function


@dataclass
class HandleMissingDataColumn:
    column_name: str
    method: str
    value: Optional[Any] = None  # Used for fillna
    kwargs: Optional[Dict[Any, Any]] = field(
        default_factory=dict
    )  # Additional Kwargs for function
