from dataclasses import dataclass, field
import json
import pprint
from typing import Any, Dict, List

import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport


@dataclass
class PandasDataFrameProfiler:
    def __init__(self, data):
        self.data: pd.DataFrame = data
        self.data_profile_dict: Dict[str, Any] = field(default_factory=dict)
        self.data_profile_df: pd.DataFrame = field(default_factory=pd.DataFrame())

    def flatten_profile_to_dataframe(self):
        def flatten_dict(d, parent_key="", sep="_"):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Handle lists (convert to string or unpack further based on your requirement)
                    items.append((new_key, ", ".join(map(str, v))))
                else:
                    items.append((new_key, v))
            return dict(items)

        flattened_data = {
            component: flatten_dict(data) if isinstance(data, dict) else data
            for component, data in self.data_profile_dict.items()
        }

        # Accumulate data in a dictionary first
        data_dict = {}
        for component, metrics in flattened_data.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        for col, val in value.items():
                            data_dict[f"{component}_{metric}_{col}"] = val
                    else:
                        data_dict[f"{component}_{metric}"] = [value] * len(
                            self.data.columns
                        )
            else:
                data_dict[component] = [metrics] * len(self.data.columns)

        # Create the DataFrame from the accumulated dictionary
        flattened_df = pd.DataFrame.from_dict(data_dict)

        self.data_profile_df = flattened_df.loc[0:0]

        return self

    def profile_data(
        self,
        include: List[Any] = [np.number],
        minimal: bool = True,
        explorative: bool = False,
        deep: bool = True,
    ):
        # Select only numeric columns for correlation analysis
        numeric_cols = self.data.select_dtypes(include=include)

        # Generate pandas profiling report
        ydata_profile_report = ProfileReport(
            self.data, minimal=minimal, explorative=explorative
        )
        ydata_profile_report_dict = json.loads(ydata_profile_report.to_json())
        # ydata_profile_report_dict = ydata_profile_report.get_description()

        descriptive_statistics = json.loads(self.data.describe().to_json())
        # Convert dtypes to a more JSON-friendly format
        data_types = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
        missing_values = json.loads(self.data.isna().sum().to_json())
        unique_values = json.loads(self.data.nunique().to_json())
        value_counts = {
            col: json.loads(self.data[col].value_counts().to_json())
            for col in self.data.columns
        }
        correlation_analysis = json.loads(numeric_cols.corr().to_json())
        memory_usage = json.loads(self.data.memory_usage(deep=deep).to_json())

        profile = {
            "descriptive_statistics": descriptive_statistics,
            "data_types": data_types,
            "missing_values": missing_values,
            "unique_values": unique_values,
            "value_counts": value_counts,
            "correlation_analysis": correlation_analysis,
            "memory_usage": memory_usage,
            "ydata_profile_report": ydata_profile_report_dict,
        }
        self.data_profile_dict = profile

        self.flatten_profile_to_dataframe()

        return self


if __name__ == "__main__":
    # Usage Example
    import numpy as np

    data = {
        "A": [1.23, 2.34, 3.45, 1.23],
        "B": [4.567672, 5, 6.78, 4.567672],
        "C": ["foo", "bar", "baz", "foo"],
        "D": ["dog", "elephant", "frog", "dog"],
        "E": [1, np.nan, 3, 1],
        "F": [4, 5, np.nan, 4],
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-01"],
    }
    df = pd.DataFrame(data)

    profiler = PandasDataFrameProfiler(df)
    profiler.profile_data()

    print("profiler")
    pprint.pprint(profiler.data_profile_dict)
    print()
    # Now you can access different parts of the profile as needed, e.g., profile["Descriptive Statistics"]

    f = open("data_profile.json", "w")
    f.write(json.dumps(profiler.data_profile_dict))
    f.close()

    print(profiler.data_profile_df)
    print()
