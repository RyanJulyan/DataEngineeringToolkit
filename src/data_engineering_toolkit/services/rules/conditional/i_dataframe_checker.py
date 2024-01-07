import string
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import pandas as pd

# Services
from data_engineering_toolkit.services.util.enforce_attributes_meta import (
    EnforceAttributesMeta,
)


class IDataFrameChecker(metaclass=EnforceAttributesMeta):
    """
    Abstract base class for DataFrame checkers.
    """

    __required_attributes__: List[str] = [
        "data",
        "custom_functions",
        "custom_rules_schema",
        "formatter",
    ]

    @abstractmethod
    def __init__(
        self,
        data: Any,
        custom_functions: Dict[str, Callable[..., Any]] = None,
        custom_rules_schema: Dict[str, Any] = None,
        formatter: string.Formatter = None,
    ):
        """
        Initialize the checker with a DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame to check.
        """
        pass

    @abstractmethod
    def check_rules(self, rules: Dict[str, Any]) -> bool:
        """
        Check the provided rules against a defined schema.

        Parameters:
            rules (Dict): The rules to check.

        Returns:
            bool: True if the rules are valid, False otherwise.
        """
        pass

    @abstractmethod
    def check(self, rules: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply the defined rules to the DataFrame and check its values.

        Parameters:
            rules (Dict): A dictionary containing the rules to apply.

        Returns:
            pd.DataFrame: The DataFrame with additional columns indicating the validation results.
        """
        pass
