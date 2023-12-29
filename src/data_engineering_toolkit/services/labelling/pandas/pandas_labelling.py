from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Union
import re

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import pandas as pd


@dataclass
class PandasDataFrameLabbeling:
    """
    A class for performing various labbeling on a pandas DataFrames in different ways.
    """

    data: pd.DataFrame
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def review(self):
        type_tags = {col: self.tag_column(df[col]) for col in df.columns}
        self.summary_df = pd.DataFrame(
            list(type_tags.items()), columns=["Column", "ConceptualType"]
        )

    def tag_column(self, column, match_percentage=0.7):
        # Handling NaN values
        valid_data = column.dropna()

        # Applying each function and counting True values
        if len(valid_data) == 0:
            return "Empty or NaN"

        true_counts = {
            "Social Security Number": self.is_ssn(valid_data).sum(),
            "SA ID Number": self.is_sa_id_number(valid_data).sum(),
            "Passport Number": self.is_passport_number(valid_data).sum(),
            "Phone Number": self.is_phone_number(valid_data).sum(),
            "Email": self.is_email(valid_data).sum(),
            "Credit Card": self.is_credit_card(valid_data).sum(),
            "Gender": self.is_gender(valid_data).sum(),
            "IP Address": self.is_ip_address(valid_data).sum(),
            "URL": self.is_url(valid_data).sum(),
            "ZipCode": self.is_zipcode(valid_data).sum(),
            "Currency": self.is_currency(valid_data).sum(),
            "City": self.is_city(valid_data).sum(),
            "State": self.is_state(valid_data).sum(),
            "Country": self.is_country(valid_data).sum(),
            "Date": self.is_date(valid_data).sum()
            if self.is_date(valid_data).any()
            else 0,
        }

        # Determining the most likely type
        most_likely_type = max(true_counts, key=true_counts.get)
        if (
            true_counts[most_likely_type] / len(valid_data) > match_percentage
        ):  # More than 50% match
            return most_likely_type
        else:
            return "Unknown"

    # Date Check
    def is_date(self, column):
        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        return column.apply(
            lambda x: bool(re.match(date_pattern, str(x))) and self.check_date(x)
        )

    def check_date(self, value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return True
        except (ValueError, TypeError):
            return False

    # Social Security Number (SSN)
    def is_ssn(self, column):
        ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
        return column.apply(lambda x: bool(re.match(ssn_pattern, str(x))))

    # South African ID Number
    def is_sa_id_number(self, column):
        sa_id_pattern = r"^\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{7}$"
        return column.apply(lambda x: bool(re.match(sa_id_pattern, str(x))))

    # Passport Number\@classmethod
    def is_passport_number(self, column):
        passport_pattern = r"^[A-Za-z0-9]{5,9}$"  # Common pattern; adjust as needed
        return column.apply(lambda x: bool(re.match(passport_pattern, str(x))))

    # Phone Number
    def is_phone_number(self, column):
        phone_pattern = (
            r"^\+?\d{1,3}\s?\(?\d{1,4}\)?\s?-?\d{1,4}\s?-?\d{1,4}\s?-?\d{1,9}$"
        )
        return column.apply(lambda x: bool(re.match(phone_pattern, str(x))))

    # Email
    def is_email(self, column):
        email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return column.apply(lambda x: bool(re.match(email_pattern, str(x))))

    # Credit Card
    def is_credit_card(self, column):
        cc_pattern = r"^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$"
        return column.apply(lambda x: bool(re.match(cc_pattern, str(x))))

    # Gender
    def is_gender(self, column):
        return column.apply(lambda x: str(x).lower() in ["male", "female", "other"])

    # IP Address
    def is_ip_address(self, column):
        ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
        return column.apply(lambda x: bool(re.match(ip_pattern, str(x))))

    # URL
    def is_url(self, column):
        url_pattern = (
            r"^(http[s]?://)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#\[\]@!$&\'()*+,;=]+$"
        )
        return column.apply(lambda x: bool(re.match(url_pattern, str(x))))

    # Zipcode
    def is_zipcode(self, column):
        zip_pattern = r"^\d{5}$"
        return column.apply(lambda x: bool(re.match(zip_pattern, str(x))))

    # Country
    def is_country(self, column):
        country_pattern = (
            r"^[A-Za-z]+(?:\s[A-Za-z]+)*$"  # Country names, no numbers/special chars
        )
        return column.apply(lambda x: bool(re.match(country_pattern, str(x))))

    # Currency
    def is_currency(self, column):
        # This is a basic check for currency symbols
        currency_pattern = r"^[\$\€\£\¥]{1}[0-9]+(\.[0-9]{2})?$"
        return column.apply(lambda x: bool(re.match(currency_pattern, str(x))))

    # State
    def is_state(self, column):
        state_pattern = r"^[A-Za-z]+(?:\s[A-Za-z]+)*$"  # State names or abbreviations, no numbers/special chars
        return column.apply(lambda x: bool(re.match(state_pattern, str(x))))

    # City
    def is_city(self, column):
        city_pattern = r"^[A-Za-z]+(?:\s[A-Za-z]+)*$"  # City names with spaces, but no numbers/special chars
        return column.apply(lambda x: bool(re.match(city_pattern, str(x))))


if __name__ == "__main__":
    # Example Usage
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
    labbeling = PandasDataFrameLabbeling(df)

    labbeling.review()
    print("summary_df")
    print(labbeling.summary_df)
    print()
