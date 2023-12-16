from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List

# Models
from de_toolkit.models.transformations.pandas.pandas_transformations_models import HandleMissingDataColumn

# Services
from de_toolkit.services.util.enforce_attributes_meta import EnforceAttributesMeta


@dataclass
class ITransformations(metaclass=EnforceAttributesMeta):
  """Abstract base class for transforming dataframes in various ways."""

  __required_attributes__: List[str] = ['data']

  @abstractmethod
  def handle_missing_values(
      self, column_methods: List[HandleMissingDataColumn]) -> ITransformations:
    """Abstract method to handle missing data of specified columns."""
    pass