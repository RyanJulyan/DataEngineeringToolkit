from abc import ABC, abstractmethod
from typing import Any, List

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker

# Services
from data_engineering_toolkit.services.util.enforce_attributes_meta import (
    EnforceAttributesMeta,
)


class IStorageService(metaclass=EnforceAttributesMeta):
    """Abstract base class for storage brokers to be called from a service."""

    storage_broker: IStorageBroker
    __required_attributes__: List[str] = ["storage_broker"]

    @abstractmethod
    def create(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def read(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def delete(self, *args, **kwargs) -> Any:
        pass
