from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import pandas as pd


class IDataFrameChecker(ABC):
    """
    Abstract base class for DataFrame checkers.
    """

    @abstractmethod
    def __init__(
        self, df: pd.DataFrame, custom_functions: Dict[str, Callable[..., Any]] = None
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
