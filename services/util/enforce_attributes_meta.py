from abc import ABCMeta
from typing import Any, List, Type


class EnforceAttributesMeta(ABCMeta):
  """
  A metaclass that enforces the presence of specific attributes in subclasses.
  """

  def __call__(cls, *args, **kwargs):
    # Create an instance (this calls __new__ and __init__)
    instance = super().__call__(*args, **kwargs)
    # Enforce required attributes
    required_attributes = getattr(cls, '__required_attributes__', [])
    for attr in required_attributes:
      if not hasattr(instance, attr):
        raise TypeError(
            f"Instance of {cls.__name__} lacks required attribute '{attr}'")
    return instance
