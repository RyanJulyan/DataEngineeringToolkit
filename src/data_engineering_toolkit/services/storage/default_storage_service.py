from dataclasses import dataclass
from typing import Any

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker

# Services
from data_engineering_toolkit.services.storage.i_storage_service import IStorageService


@dataclass
class DefaultSourceService(IStorageService):
  storage_broker: IStorageBroker

  def create(self, **create_kwargs) -> Any:
    return self.storage_broker.create(**create_kwargs)

  def read(self, **read_kwargs) -> Any:
    return self.storage_broker.read(**read_kwargs)

  def update(self, **update_kwargs) -> Any:
    return self.storage_broker.update(**update_kwargs)

  def delete(self, **delete_kwargs) -> Any:
    return self.storage_broker.delete(**delete_kwargs)
