from abc import ABCMeta
from typing import Any, List, Type

class EnforceAttributesMeta(ABCMeta):
  """
  A metaclass that enforces the presence of specific attributes in subclasses.
  """

  def __init__(cls: Type[Any], name: str, bases: tuple, dct: dict):
    """
    Initialize the class instance and enforce required attributes.

    Args:
        cls: The class being initialized.
        name: The name of the class.
        bases: A tuple containing the base classes.
        dct: A dictionary of the class's namespace.
    """
    super().__init__(name, bases, dct)
    # Skip check for abstract classes
    if not getattr(cls, '__abstractmethods__', False):
      required_attributes: List[str] = getattr(cls, '__required_attributes__', [])
      for attr in required_attributes:
        if not hasattr(cls, attr):
          raise TypeError(f"Class {name} lacks required attribute '{attr}'")