from abc import ABC, abstractmethod
from typing import Any


class IStorageBroker(ABC):
    """A Base class interface definition for data storage insertion and retrieval.

    Args:
        ABC: Abstract Base Class (Interface) in python.
    """

    @abstractmethod
    def create(self, *args, **kwargs) -> Any:
        """Create records."""
        pass

    @abstractmethod
    def read(self, *args, **kwargs) -> Any:
        """Read records."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Update records."""
        pass

    @abstractmethod
    def delete(self, *args, **kwargs) -> Any:
        """Delete records."""
        pass
