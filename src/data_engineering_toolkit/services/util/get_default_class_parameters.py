import inspect
from typing import Any, Dict, Type


def get_default_class_parameters(class_type: Type[Any],
                                 method_name: str = "__init__"
                                 ) -> Dict[str, Any]:
  """Gets default parameters for a specified method of a given class.

    Args:
        class_type (class): The class for which to get the method's default parameters.
        method_name (str): The method name. Defaults to "__init__".

    Returns:
        dict: A dictionary of parameter names and their default values.

    Raises:
        ValueError: If the specified method is not found in the given class.

    Example:
        >>> from sklearn.neighbors import KNeighborsRegressor
        >>> print(get_default_parameters(KNeighborsRegressor))
    """
  # Retrieve the method
  method = getattr(class_type, method_name, None)
  if not method:
    raise ValueError(
        f"Method {method_name} not found in class {class_type.__name__}.")

  # Retrieve the signature of the method
  method_signature = inspect.signature(method)

  # Extract the default values for all parameters
  default_parameters = {
      k: v.default
      for k, v in method_signature.parameters.items()
      if v.default is not inspect.Parameter.empty
  }

  return default_parameters
