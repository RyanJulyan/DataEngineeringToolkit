from __future__ import annotations

from abc import ABC, abstractmethod
import dis
from types import SimpleNamespace
from typing import Any, List, Optional

import pandas as pd

NoneType = type(None)


class BaseComputeRule(ABC):
    """
    A base class for defining rules in a rule-based compute system.

    This class provides a template for creating rules, including attributes for rule name, priority, dependencies,
    order of addition, and additional arguments. It also includes methods for comparing rules, adding dependencies,
    and checking types of data.

    Attributes:
        name (str): The name of the rule.
        priority (int): The priority of the rule, used for ordering.
        depends_on (List[BaseComputeRule]): A list of other rules that this rule depends on.
        addition_order (int): The order in which the rule was added to the registry.
        args (SimpleNamespace): A namespace for storing additional arguments.

    Methods:
        add_dependency(rule: BaseComputeRule): Adds a dependency to the rule.
        apply(data: Any): Abstract method to apply the rule to the given data.
        validate(data: Any): Abstract method to validate the input data.
        check_apply_has_return() -> bool: Checks if the apply method has a return statement.
        is_int(data: Any) -> bool: Checks if the data is an integer.
        is_float(data: Any) -> bool: Checks if the data is a float.
        is_str(data: Any) -> bool: Checks if the data is a string.
        is_list(data: Any) -> bool: Checks if the data is a list.
        is_pandas_dataframe(data: Any) -> bool: Checks if the data is a Pandas DataFrame.
    """

    def __init__(
        self,
        name: str,
        priority: int = 0,
        depends_on: Optional[List[BaseComputeRule]] = None,
        addition_order: int = 0,
        **kwargs: Any,
    ):
        """
        Initializes a new instance of the BaseComputeRule class.

        Args:
            name (str): The name of the rule.
            priority (int, optional): The priority of the rule. Defaults to 0.
            depends_on (Optional[List[BaseComputeRule]], optional): Rules that this rule depends on. Defaults to None.
            addition_order (int): The order that the Rule was added. Defaults to 0.
            **kwargs: Additional keyword arguments.
        """

        self.name: str = name
        self.priority: int = priority
        self.depends_on: List[BaseComputeRule] = (
            depends_on if depends_on is not None else []
        )
        self.addition_order: int = addition_order
        self.args: SimpleNamespace = SimpleNamespace(
            **kwargs
        )  # Store additional arguments as attributes under args

    def __lt__(self, other: BaseComputeRule) -> bool:
        """Less than comparison for sorting rules in the priority queue.

        Args:
            other (BaseComputeRule): The other rule to compare to.

        Returns:
            bool: True if this rule has a higher priority or was added earlier than the other.
        """
        if self.priority == other.priority:
            return self.addition_order < other.addition_order
        return self.priority > other.priority

    def add_dependency(self, rule: BaseComputeRule) -> None:
        """Add a dependency to the rule.

        Args:
            rule (BaseComputeRule): The rule to be added as a dependency.
        """
        if rule not in self.depends_on:
            self.depends_on.append(rule)

    @abstractmethod
    def apply(self, data: Any) -> Any:
        """Apply the rule to the given data. This method should be overridden.

        Args:
            data (Any): The data to which the rule is applied.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.

        Returns:
            Any: The result of applying the rule to the data.
        """

        raise NotImplementedError("Each rule must implement an apply method")

    @abstractmethod
    def validate(self, data: Any) -> Any:
        """Use this to validate the input data has what you need to apply the function.

        Args:
            data (Any): The data to which the rule is applied.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.

        Returns:
            Any: The result of applying the validation to the data.
        """
        raise NotImplementedError("Each rule must implement a Validate method")

    @classmethod
    def check_apply_has_return(cls) -> bool:
        """Check if the apply method has a return statement.

        Returns:
            bool: True if the apply method has a return statement, False otherwise.
        """

        instructions = dis.get_instructions(cls.apply)
        return any(instr.opname == "RETURN_VALUE" for instr in instructions)

    def is_str(self, data):
        """Check if the data is a string."""
        return isinstance(data, str)

    def is_int(self, data):
        """Check if the data is an integer."""
        return isinstance(data, int)

    def is_float(self, data):
        """Check if the data is a float."""
        return isinstance(data, float)

    def is_complex(self, data):
        """Check if the data is a complex."""
        return isinstance(data, complex)

    def is_list(self, data):
        """Check if the data is a list."""
        return isinstance(data, list)

    def is_tuple(self, data):
        """Check if the data is a tuple."""
        return isinstance(data, tuple)

    def is_range(self, data):
        """Check if the data is a range."""
        return isinstance(data, range)

    def is_dict(self, data):
        """Check if the data is a dict."""
        return isinstance(data, dict)

    def is_set(self, data):
        """Check if the data is a set."""
        return isinstance(data, set)

    def is_frozenset(self, data):
        """Check if the data is a frozenset."""
        return isinstance(data, frozenset)

    def is_bool(self, data):
        """Check if the data is a bool."""
        return isinstance(data, bool)

    def is_bytes(self, data):
        """Check if the data is a bytes."""
        return isinstance(data, bytes)

    def is_bytearray(self, data):
        """Check if the data is a bytearray."""
        return isinstance(data, bytearray)

    def is_memoryview(self, data):
        """Check if the data is a memoryview."""
        return isinstance(data, memoryview)

    def is_none_type(self, data):
        """Check if the data is a NoneType."""
        return isinstance(data, NoneType)

    def is_pandas_dataframe(self, data):
        """Check if the data is a Pandas DataFrame."""
        return isinstance(data, pd.DataFrame)

    def is_pandas_series(self, data):
        """Check if the data is a Pandas Series."""
        return isinstance(data, pd.Series)
