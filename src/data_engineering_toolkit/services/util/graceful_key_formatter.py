import string
import re
from typing import Any, List, Dict


class GracefulKeyFormatter(string.Formatter):
    """
    A custom string.Formatter to handle missing parameters in the formatter more gracefully
    """

    def get_value(self, key: str, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        """
        Retrieve the value of the given key from the provided keyword arguments.

        Parameters:
            key (str): The key to look up.
            args (List[Any]): Positional arguments (unused in this method).
            kwargs (Dict[str, Any]): Keyword arguments containing the values for formatting.

        Returns:
            Any: The value associated with the given key or an empty string if the key is not found.
        """
        return kwargs.get(key, "")

    def vformat(
        self, format_string: str, args: List[Any], kwargs: Dict[str, Any]
    ) -> str:
        """
        Format the given format string using the provided arguments and post-process the result.

        Parameters:
            format_string (str): The format string containing placeholders.
            args (List[Any]): Positional arguments for formatting.
            kwargs (Dict[str, Any]): Keyword arguments for formatting.

        Returns:
            str: The formatted string with any sequence of multiple underscores replaced by a single underscore and stripped of leading and trailing underscores.
        """
        result = super().vformat(format_string, args, kwargs)
        # Remove any sequence of multiple underscores
        result = re.sub(r"_+", "_", result)
        # Strip leading and trailing underscores
        return result.strip("_")
