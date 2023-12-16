from abc import ABCMeta
from typing import Any, List, Type


class EnforceAttributesMeta(ABCMeta):
  """
    A metaclass that enforces the presence of specific instance attributes in subclasses.

    This metaclass extends ABCMeta, allowing it to be used with abstract base classes. 
    It overrides the __call__ method to check if instances of subclasses have defined 
    all required attributes specified in the '__required_attributes__' class attribute.
    """

  def __call__(cls: Type[Any], *args: Any, **kwargs: Any) -> Any:
    """
        Create an instance of the class and enforce the presence of required attributes.

        This method is invoked when an instance of a class using this metaclass is created.
        It first calls the parent class's __call__ method to create an instance and then
        checks if this instance has all the attributes listed in '__required_attributes__'.

        Args:
            cls: The class being instantiated.
            *args: Variable length argument list for the class constructor.
            **kwargs: Arbitrary keyword arguments for the class constructor.

        Returns:
            An instance of 'cls'.

        Raises:
            TypeError: If any required attribute is missing in the created instance.
        """
    # Create an instance (this calls __new__ and __init__)
    instance = super().__call__(*args, **kwargs)

    # Enforce required attributes
    required_attributes = getattr(cls, '__required_attributes__', [])
    for attr in required_attributes:
      if not hasattr(instance, attr):
        raise TypeError(
            f"Instance of {cls.__name__} lacks required attribute '{attr}'")

    return instance
