from functools import wraps
from typing import Callable, Any, get_type_hints, TypeVar

# Creating a TypeVar for generic function types
F = TypeVar('F', bound=Callable[..., Any])


def enforce_function_signature(
    target_signature: Callable[..., Any]) -> Callable[[F], F]:
  """Decorator to enforce that the decorated function matches the specified signature.

    Args:
        target_signature (Callable[..., Any]): The function whose signature is to be enforced.

    Returns:
        Callable[[F], F]: A decorator that takes a function and returns a function.
    """
  target_signature_hints = get_type_hints(target_signature)

  def decorator(func: F) -> F:
    """Decorator that checks the signature of the given function.

        Args:
            func (F): The function to check.

        Returns:
            F: The original function, if it matches the target signature.

        Raises:
            TypeError: If the argument types or return type do not match the target signature.
        """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
      """Wrapper function that performs the signature check.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                Any: The result of the function call, if the signature matches.

            Raises:
                TypeError: If the argument types or return type do not match the target signature.
            """
      func_signature_hints = get_type_hints(func)

      # Check return type
      if func_signature_hints.get('return') != target_signature_hints.get(
          'return'):
        raise TypeError(
            f"Return type must be {target_signature_hints.get('return')}")

      # Check arguments
      for name, type_ in target_signature_hints.items():
        if name == 'return':
          continue
        if name not in func_signature_hints:
          raise TypeError(f"Argument '{name}' must be present")
        if name in func_signature_hints and func_signature_hints[name] != type_:
          raise TypeError(f"Argument '{name}' must be of type {type_}")

      return func(*args, **kwargs)

    return wrapper  # type: ignore

  return decorator
