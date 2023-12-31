import string
from typing import Any, Callable, Dict

import pandas as pd
from jsonschema import Draft7Validator, exceptions

from data_engineering_toolkit.services.rules.conditional.i_dataframe_checker import (
    IDataFrameChecker,
)
from data_engineering_toolkit.services.util.graceful_key_formatter import (
    GracefulKeyFormatter,
)


class PandasDataFrameChecker(IDataFrameChecker):
    """
    A class to check and check pandas DataFrame based on custom and default functions.
    """

    @staticmethod
    def is_number(value: Any) -> bool:
        """
        Check if the given value is a number.

        Parameters:
            value (Any): The value to check.

        Returns:
            bool: True if the value is a number, False otherwise.
        """
        return isinstance(value, (int, float))

    @staticmethod
    def is_string(value: Any) -> bool:
        """
        Check if the given value is a string.

        Parameters:
            value (Any): The value to check.

        Returns:
            bool: True if the value is a string, False otherwise.
        """
        return isinstance(value, str)

    @classmethod
    def default_functions(cls) -> Dict[str, Callable[..., Any]]:
        """
        Return a dictionary of default functions.

        Returns:
            Dict[str, Callable[..., Any]]: A dictionary of function names mapped to their corresponding static methods.
        """
        return {
            "is_number": cls.is_number,
            "is_string": cls.is_string,
            # Add more default functions as needed
        }

    DEFAULT_FUNCTIONS = {
        "is_number": is_number,
        "is_string": is_string,
        # Add more default functions as needed
    }

    DEFAULT_RULES_SCHEMA = {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "column": {"type": "string"},
                        "function": {"type": "string"},
                        "fact": {"type": "boolean"},
                        "format": {"type": "string"},
                        "format_kwargs": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["name", "column", "function", "fact"],
                    "additionalProperties": False,  # This ensures that additional properties cause a check error
                },
            }
        },
        "required": ["rules"],
    }

    def __init__(
        self,
        df: pd.DataFrame,
        custom_functions: Dict[str, Callable[..., Any]] = None,
        custom_rules_schema: Dict[str, Any] = None,
        formatter: string.Formatter = None,
    ):
        """
        Initialize the PandasDataFrameChecker with a DataFrame, custom functions, and a formatter.

        Parameters:
            df (pd.DataFrame): The DataFrame to check.
            custom_functions (Dict[str, Callable[..., Any]], optional): A dictionary of custom functions. Defaults to None.
            formatter (string.Formatter, optional): A custom formatter for column names. Defaults to GracefulKeyFormatter.
        """
        self.df = df
        self.formatter = formatter or GracefulKeyFormatter()
        self.rules_schema = custom_rules_schema or self.DEFAULT_RULES_SCHEMA

        # Merge default functions with custom functions
        self.functions = {**self.DEFAULT_FUNCTIONS, **(custom_functions or {})}

    def check_rules(self, rules: Dict) -> bool:
        """
        Check the provided rules against the defined self.rules_schema.

        Parameters:
            rules (Dict): The rules to check.

        Returns:
            bool: True if the rules are checked, False otherwise.

        Raises:
            jsonschema.exceptions.ValidationError: If the rules do not match the schema.
        """
        validator = Draft7Validator(self.rules_schema)
        errors = [error.message for error in validator.iter_errors(rules)]
        if errors:
            error_messages = "\n".join(errors)
            raise exceptions.ValidationError(
                f"Provided rules do not match the expected schema. Errors:\n{error_messages}"
            )
        return True

    def check(self, rules: Dict) -> pd.DataFrame:
        """
        Check_rules to rules_schema
        Apply the defined rules to the DataFrame and check its values.

        Parameters:
            rules (Dict): A dictionary containing the rules to apply.

        Returns:
            pd.DataFrame: The DataFrame with additional columns indicating the check results.
        """

        self.check_rules(rules=rules)
        for rule in rules["rules"]:
            column = rule["column"]
            function_name = rule["function"]
            rule_name = rule["name"]
            fact = rule["fact"]
            format_str = rule.get("format", "{rule_name}_check")
            format_kwargs = rule.get("format_kwargs", {})
            format_kwargs["rule_name"] = rule_name

            function = self.functions.get(function_name)
            if not function:
                raise ValueError(f"Function '{function_name}' not found.")

            column_name = self.formatter.format(format_str, **format_kwargs)
            self.df[column_name] = self.df[column].apply(
                self._apply_function_wrapper(function, fact)
            )

        return self.df

    @staticmethod
    def _apply_function_wrapper(
        function: Callable[..., Any], fact: Any
    ) -> Callable[..., int]:
        """
        Return a wrapper function to apply the given function and compare its result with the expected fact.

        Parameters:
            function (Callable[..., Any]): The function to apply.
            fact (Any): The expected result.

        Returns:
            Callable[..., int]: The wrapper function.
        """

        def wrapper(value: Any) -> int:
            try:
                result = function(value)
                if result == fact:
                    return 1
                else:
                    return 0
            except Exception:
                return -1

        return wrapper


# Usage:
if __name__ == "__main__":
    # Define custom function
    def is_uppercase(value: Any) -> bool:
        """
        Check if the given value, when converted to a string, is in uppercase.

        Parameters:
            value (Any): The value to check.

        Returns:
            bool: True if the value is in uppercase, False otherwise.
        """
        return str(value).isupper()

    # Define your rules
    rules = {
        "rules": [
            {
                "name": "age_is_number",
                "column": "age",
                "function": "is_number",
                "fact": True,
                "format": "{rule_name}_{suffix}__{test}",
                "format_kwargs": {
                    "suffix": "check_result",
                },
            },
            {
                "name": "name_is_string",
                "column": "name",
                "function": "is_string",
                "fact": True,
            },
            {
                "name": "name_is_uppercase",
                "column": "name",
                "function": "is_uppercase",
                "fact": True,
            },
        ],
    }

    data = pd.DataFrame(
        [
            {"name": "John", "age": 30},
            {"name": "Alice", "age": "thirty"},  # This will cause a check error
            {"name": "Bob", "age": 40},
            {"name": "bob lowwer", "age": "250"},
        ]
    )

    checker = PandasDataFrameChecker(
        data, custom_functions={"is_uppercase": is_uppercase}
    )
    checked_data = checker.check(rules)

    print(checked_data)

    # Define your invalid rules
    invalid_rules = {
        "rules": [
            {
                "invalid_key": "value1",
                "name": "age_is_number",
                "column": "age",
                "function": "is_number",
                "fact": True,
                "format": "{rule_name}_{suffix}__{test}",
                "format_kwargs": {
                    "suffix": "check_result",
                },
            },
            {
                "another_invalid_key": "value1",
                "name": "name_is_string",
                "column": "name",
                "function": "is_string",
                "fact": True,
            },
            {
                "name": "name_is_uppercase",
                "column": "name",
                "function": "is_uppercase",
                "fact": True,
            },
        ],
    }

    checker.check(rules=invalid_rules)
    # checker.check_rules(rules=invalid_rules)
