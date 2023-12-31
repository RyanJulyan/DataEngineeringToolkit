from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd
from jsonschema import Draft7Validator
from jsonschema.protocols import Validator


class PandasDataFrameValidator:
    def __init__(
        self,
        data: pd.DataFrame,
        schema: Dict[str, Any],
        validator: Optional[Validator] = None,
    ):
        """
        Initialize the DataFrameValidator with a DataFrame and a JSON schema.

        Parameters:
            data (pd.DataFrame): The DataFrame to validate.
            schema (Dict): The JSON schema to validate against.
            validator (Optional[Validator]): The JSON schema validator to validate.
        """
        self.data: pd.DataFrame = data
        self.schema: Dict[str, Any] = schema
        self.validator: Validator = validator
        if validator is None:
            self.validator = Draft7Validator(schema)
        else:
            self.validator = self.validator(schema)

        self.validate()

    @property
    def valid_data(self) -> pd.DataFrame:
        """
        Get the valid data from the DataFrame after validate method is run.

        Returns:
            pd.DataFrame: The valid data.
        """
        valid_data = self.data[self.data["is_valid"]]
        return valid_data.drop(columns=["is_valid", "error_message"])

    @property
    def invalid_data(self) -> pd.DataFrame:
        """Get the invalid data from the DataFrame."""
        invalid_data = self.data[~self.data["is_valid"]]
        return invalid_data.drop(columns="is_valid")

    def validate(self) -> DataFrameValidator:
        """
        Apply JSON schema validation to a DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with additional columns that indicate whether each row is valid and the error message if not.
        """

        def get_validation_result(row: pd.Series) -> Dict[str, Union[bool, str]]:
            errors = list(self.validator.iter_errors(row.to_dict()))
            if errors:
                return {"is_valid": False, "error_message": errors[0].message}
            else:
                return {"is_valid": True, "error_message": None}

        validation_results = self.data.apply(
            get_validation_result, axis=1, result_type="expand"
        )
        self.data = pd.concat([self.data, validation_results], axis=1)

        return self


if __name__ == "__main__":
    # Define your data
    data = pd.DataFrame(
        [
            {"name": "John", "age": 30},
            {"name": "Alice", "age": "thirty"},  # This will cause a validation error
            {"name": "Bob", "age": 40},
        ]
    )

    # Define your schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
    }

    # Create an instance of PandasDataFrameValidator
    validator = PandasDataFrameValidator(data, schema)

    # Get valid and invalid data
    # valid_data = validator.get_valid_data()
    # invalid_data = validator.get_invalid_data()

    # Display the results
    print("\nAll data:")
    print(validator.data)
    print("\nValid data:")
    print(validator.valid_data)
    print("\nInvalid data:")
    print(validator.invalid_data)
