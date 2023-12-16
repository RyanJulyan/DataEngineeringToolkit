from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class HandleMissingDataColumn:
    column_name: str
    method: str
    value: Optional[Any] = None  # Used for fillna
    kwargs: Optional[Dict[Any, Any]] = field(default_factory=dict)  # Additional Kwargs for function
