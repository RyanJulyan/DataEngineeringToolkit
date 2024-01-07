from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Brokers
from data_engineering_toolkit.brokers.storage.i_storage_broker import IStorageBroker
from data_engineering_toolkit.brokers.storage.pandas.pandas_csv_storage_broker import (
    PandasCSVStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_excel_storage_broker import (
    PandasExcelStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_feather_storage_broker import (
    PandasFeatherStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_fwf_storage_broker import (
    PandasFWFStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_gbq_storage_broker import (
    PandasGBQStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_hdf5_storage_broker import (
    PandasHDF5StorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_html_storage_broker import (
    PandasHTMLStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_json_storage_broker import (
    PandasJSONStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_latex_storage_broker import (
    PandasLaTeXStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_orc_storage_broker import (
    PandasORCStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_parquet_storage_broker import (
    PandasParquetStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_sas_storage_broker import (
    PandasSASStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_sql_storage_broker import (
    PandasSQLStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_stata_storage_broker import (
    PandasStataStorageBroker,
)
from data_engineering_toolkit.brokers.storage.pandas.pandas_xml_storage_broker import (
    PandasXMLStorageBroker,
)


@dataclass
class PandasStorageBroker(IStorageBroker):
    """
    Pandas data storage insertion and retrieval.
    """

    storage_type: Dict[str, IStorageBroker] = field(default_factory=dict)

    def __init__(
        self,
        sql_alch_engine: Any = None,
        google_project_id: Any = None,
        custom_storage_types: Dict[str, IStorageBroker] = {},
    ):
        self.storage_type: Dict[str, IStorageBroker] = {
            "csv": PandasCSVStorageBroker(),
            "excel": PandasExcelStorageBroker(),
            "feather": PandasFeatherStorageBroker(),
            "fwf": PandasFWFStorageBroker(),
            "gbq": PandasGBQStorageBroker(project_id=google_project_id),
            "hdf5": PandasHDF5StorageBroker(),
            "html": PandasHTMLStorageBroker(),
            "json": PandasJSONStorageBroker(),
            "latex": PandasLaTeXStorageBroker(),
            "orc": PandasORCStorageBroker(),
            "parquet": PandasParquetStorageBroker(),
            "sas": PandasSASStorageBroker(),
            "sql": PandasSQLStorageBroker(engine=sql_alch_engine),
            "strata": PandasStataStorageBroker(),
            "xml": PandasXMLStorageBroker(),
            **custom_storage_types,
        }

    def create(self, format: str, *args: Any, **kwargs: Any) -> Any:
        """Create records."""
        try:
            self.storage_type[format].create(*args, **kwargs)
        except KeyError as e:
            raise KeyError(
                f"Unsupported Format: {format}! availible formats: {self.storage_type.keys()}"
            )
        except Exception as e:
            raise Exception(e)

    def read(self, format: str, *args: Any, **kwargs: Any) -> Any:
        """Read records."""
        try:
            self.storage_type[format].read(*args, **kwargs)
        except KeyError as e:
            raise KeyError(
                f"Unsupported Format: {format}! availible formats: {self.storage_type.keys()}"
            )
        except Exception as e:
            raise Exception(e)

    def update(self, format: str, *args: Any, **kwargs: Any) -> Any:
        """Update records."""
        try:
            self.storage_type[format].update(*args, **kwargs)
        except KeyError as e:
            raise KeyError(
                f"Unsupported Format: {format}! availible formats: {self.storage_type.keys()}"
            )
        except Exception as e:
            raise Exception(e)

    def delete(self, format: str, *args: Any, **kwargs: Any) -> Any:
        """Delete records."""
        try:
            self.storage_type[format].delete(*args, **kwargs)
        except KeyError as e:
            raise KeyError(
                f"Unsupported Format: {format}! availible formats: {self.storage_type.keys()}"
            )
        except Exception as e:
            raise Exception(e)
