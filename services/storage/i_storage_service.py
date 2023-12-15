from abc import ABC, abstractmethod
from typing import Any

from brokers.storage.i_storage_broker import IStorageBroker


class IStorageService(ABC):
    storage_broker: IStorageBroker

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
