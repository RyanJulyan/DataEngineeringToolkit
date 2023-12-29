from __future__ import annotations

from dataclasses import dataclass
import string
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.utils import resample
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import numpy as np


@dataclass
class OneHotEncoderMethod:
    method: str = "onehot"
    encoder: OneHotEncoder = OneHotEncoder()


@dataclass
class LabelEncoderMethod:
    method: str = "label"
    encoder: LabelEncoder = LabelEncoder()


@dataclass
class PandasDataFrameCleaner:
    data: pd.DataFrame

    def drop_missing_values(self) -> PandasDataFrameCleaner:
        """
        Drops all rows in the DataFrame containing missing values.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe.dropna(inplace=True)

        return self

    def fill_missing_values(
        self, value: Union[str, int, float]
    ) -> PandasDataFrameCleaner:
        """
        Fills missing values in the DataFrame with a specified value.

        Args:
            value (Union[str, int, float]): The value used to fill missing values.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe.fillna(value, inplace=True)

        return self

    def remove_duplicates(self) -> PandasDataFrameCleaner:
        """
        Removes duplicate rows from the DataFrame.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe.drop_duplicates(inplace=True)

        return self

    def string_lower(self, column: str) -> PandasDataFrameCleaner:
        """
        Converts all characters in a specified column to lowercase.

        Args:
            column (str): The name of the column to be converted to lowercase.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.lower()

        return self

    def string_upper(self, column: str) -> PandasDataFrameCleaner:
        """
        Converts all characters in a specified column to uppercase.

        Args:
            column (str): The name of the column to be converted to uppercase.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.upper()

        return self

    def string_title_case(self, column: str) -> PandasDataFrameCleaner:
        """
        Converts the first character of each word in a specified column to uppercase.

        Args:
            column (str): The name of the column to be converted to title case.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.title()

        return self

    def string_capitalize(self, column: str) -> PandasDataFrameCleaner:
        """
        Capitalizes the first character of the first word in a specified column.

        Args:
            column (str): The name of the column to be capitalized.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.capitalize()

        return self

    def string_swapcase(self, column: str) -> PandasDataFrameCleaner:
        """
        Swaps the case of each character in a specified column.

        Args:
            column (str): The name of the column where the case of each character is to be swapped.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.swapcase()

        return self

    def string_casefold(self, column: str) -> PandasDataFrameCleaner:
        """
        Converts the specified column to casefolded (lowercase) strings, used for caseless matching.

        Args:
            column (str): The name of the column to casefold.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.casefold()

        return self

    def string_strip(self, column: str, chars: str = None) -> PandasDataFrameCleaner:
        """
        Strips leading and trailing characters in a specified column.

        Args:
            column (str): The name of the column to strip characters from.
            chars (str, optional): The set of characters to remove. If None, whitespaces are removed.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.strip(to_strip=chars)

        return self

    def string_left_strip(
        self, column: str, chars: str = None
    ) -> PandasDataFrameCleaner:
        """
        Strips leading characters (from the left side) in a specified column.

        Args:
            column (str): The name of the column to strip characters from.
            chars (str, optional): The set of characters to remove. If None, whitespaces are removed.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.lstrip(to_strip=chars)

        return self

    def string_right_strip(
        self, column: str, chars: str = None
    ) -> PandasDataFrameCleaner:
        """
        Strips trailing characters (from the right side) in a specified column.

        Args:
            column (str): The name of the column to strip characters from.
            chars (str, optional): The set of characters to remove. If None, whitespaces are removed.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.rstrip(to_strip=chars)

        return self

    def string_split(
        self, column: str, pat: str = ",", expand: bool = True
    ) -> PandasDataFrameCleaner:
        """
        Splits the string in a specified column by a given pattern.

        Args:
            column (str): The name of the column to split.
            pat (str, optional): The pattern or delimiter to split by. Default is ','.
            expand (bool, optional): Whether to expand the split strings into separate columns. Default is True.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.split(
            pat=pat, expand=expand
        )

        return self

    def string_remove_prefix(self, column: str, prefix: str) -> PandasDataFrameCleaner:
        """
        Removes a specified prefix from the start of the string in a given column.

        Args:
            column (str): The name of the column.
            prefix (str): The prefix to remove.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        str_series = self.dataframe[column].astype(str).str
        if hasattr(str_series, "removeprefix"):  # Python 3.9 and newer
            self.dataframe[column] = str_series.removeprefix(prefix)
        else:
            self.dataframe[column] = str_series.apply(
                lambda x: x[len(prefix) :] if x.startswith(prefix) else x
            )

        return self

    def string_remove_suffix(self, column: str, suffix: str) -> PandasDataFrameCleaner:
        """
        Removes a specified suffix from the end of the string in a given column.

        Args:
            column (str): The name of the column.
            suffix (str): The suffix to remove.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        str_series = self.dataframe[column].astype(str).str
        if hasattr(str_series, "removesuffix"):  # Python 3.9 and newer
            self.dataframe[column] = str_series.removesuffix(suffix)
        else:
            self.dataframe[column] = str_series.apply(
                lambda x: x[: -len(suffix)] if x.endswith(suffix) else x
            )

        return self

    def string_pad(
        self,
        column: str,
        width: int,
        side: Literal["both", "left", "right"] = "both",
        fillchar: str = " ",
    ) -> PandasDataFrameCleaner:
        """
        Pads the string in the specified column to a certain width.

        Args:
            column (str): The name of the column.
            width (int): The final width of the string.
            side (Literal["both", "left", "right"], optional): The side to pad on. Default is "both".
            fillchar (str, optional): The character used for padding. Default is a space.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        if side == "left":
            self.dataframe[column] = self.dataframe[column].str.pad(
                width, side="left", fillchar=fillchar
            )
        elif side == "right":
            self.dataframe[column] = self.dataframe[column].str.pad(
                width, side="right", fillchar=fillchar
            )
        else:  # default to both sides
            self.dataframe[column] = self.dataframe[column].str.pad(
                width, side="both", fillchar=fillchar
            )

        return self

    def string_center(
        self, column: str, width: int, fillchar: str = " "
    ) -> PandasDataFrameCleaner:
        """
        Centers the string in the specified column.

        Args:
            column (str): The name of the column.
            width (int): The width to center the string within.
            fillchar (str, optional): The character used for padding. Default is a space.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.center(width, fillchar)

        return self

    def string_ljust(
        self, column: str, width: int, fillchar: str = " "
    ) -> PandasDataFrameCleaner:
        """
        Left-justifies the string in the specified column.

        Args:
            column (str): The name of the column.
            width (int): The width to justify the string within.
            fillchar (str, optional): The character used for padding. Default is a space.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.ljust(width, fillchar)

        return self

    def string_rjust(
        self, column: str, width: int, fillchar: str = " "
    ) -> PandasDataFrameCleaner:
        """
        Right-justifies the string in the specified column.

        Args:
            column (str): The name of the column.
            width (int): The width to justify the string within.
            fillchar (str, optional): The character used for padding. Default is a space.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.rjust(width, fillchar)

        return self

    def string_zfill(self, column: str, width: int) -> PandasDataFrameCleaner:
        """
        Pads the string in the specified column on the left with zeros until it reaches the specified width.

        Args:
            column (str): The name of the column.
            width (int): The final width of the string.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.zfill(width)

        return self

    def string_slice(
        self, column: str, start: int = None, stop: int = None
    ) -> PandasDataFrameCleaner:
        """
        Slices each string in the specified column.

        Args:
            column (str): The name of the column.
            start (int, optional): The starting index position for the slice. Default is None.
            stop (int, optional): The ending index position for the slice. Default is None.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.slice(start, stop)

        return self

    def string_slice_replace(
        self, column: str, start: int = None, stop: int = None, repl: str = ""
    ) -> PandasDataFrameCleaner:
        """
        Replaces a slice of each string in the specified column.

        Args:
            column (str): The name of the column.
            start (int, optional): The starting index position for the slice. Default is None.
            stop (int, optional): The ending index position for the slice. Default is None.
            repl (str, optional): The replacement string. Default is an empty string.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.slice_replace(
            start, stop, repl
        )

        return self

    def string_extract(
        self, column: str, regex: str, expand: bool = True
    ) -> PandasDataFrameCleaner:
        """
        Extracts a part of the string in the specified column based on a regular expression.

        Args:
            column (str): The name of the column.
            regex (str): The regular expression pattern to use for extraction.
            expand (bool, optional): Whether to expand the results into separate columns. Default is True.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.extract(
            regex, expand=expand
        )

        return self

    def string_extract_all(self, column: str, regex: str) -> PandasDataFrameCleaner:
        """
        Extracts all occurrences of a pattern in the specified column based on a regular expression.

        Args:
            column (str): The name of the column.
            regex (str): The regular expression pattern to use for extraction.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.extractall(regex)

        return self

    def string_encode(
        self, column: str, encoding: str = "utf-8"
    ) -> PandasDataFrameCleaner:
        """
        Encodes the strings in the specified column to the given encoding.

        Args:
            column (str): The name of the column.
            encoding (str, optional): The encoding to use. Default is 'utf-8'.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.encode(encoding)

        return self

    def string_decode(
        self, column: str, encoding: str = "utf-8"
    ) -> PandasDataFrameCleaner:
        """
        Decodes the strings in the specified column from the given encoding.

        Args:
            column (str): The name of the column.
            encoding (str, optional): The encoding to decode from. Default is 'utf-8'.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.decode(encoding)

        return self

    def string_get_dummies(
        self, column: str, separator: str = "|"
    ) -> PandasDataFrameCleaner:
        """
        Converts categorical variable(s) into dummy/indicator variables.

        Args:
            column (str): The name of the column to convert.
            separator (str, optional): The character used to split the categories. Default is '|'.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        dummies = self.dataframe[column].str.get_dummies(sep=separator)
        self.dataframe = pd.concat([self.dataframe, dummies], axis=1)

        return self

    def string_remove_punctuation(self, column: str) -> PandasDataFrameCleaner:
        """
        Removes all punctuation from the string in the specified column.

        Args:
            column (str): The name of the column from which to remove punctuation.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.replace(
            f"[{string.punctuation}]", "", regex=True
        )

        return self

    def string_remove_whitespace(self, column: str) -> PandasDataFrameCleaner:
        """
        Removes leading and trailing whitespaces from the string in the specified column.

        Args:
            column (str): The name of the column from which to remove whitespaces.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.strip()

        return self

    def string_remove_extra_spaces(self, column: str) -> PandasDataFrameCleaner:
        """
        Removes extra spaces from the string in the specified column, leaving only single spaces between words.

        Args:
            column (str): The name of the column from which to remove extra spaces.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.replace(
            r"\s+", " ", regex=True
        )

        return self

    def string_replace_specific_words(
        self, column: str, word_map: Dict[str, str]
    ) -> PandasDataFrameCleaner:
        """
        Replaces specific words in the string of the specified column based on a mapping dictionary.

        Args:
            column (str): The name of the column in which to replace words.
            word_map (Dict[str, str]): A dictionary mapping words to their replacements.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        for old_word, new_word in word_map.items():
            self.dataframe[column] = self.dataframe[column].str.replace(
                old_word, new_word, regex=True
            )

        return self

    def string_translate(
        self, column: str, translation_table: Dict[int, Union[int, str, None]]
    ) -> PandasDataFrameCleaner:
        """
        Translates characters in the specified column based on a translation table.

        Args:
            column (str): The name of the column.
            translation_table (Dict[int, Union[int, str, None]]): The translation table specifying the conversion.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.translate(translation_table)

        return self

    def string_normalize(
        self, column: str, form: str = "NFC"
    ) -> PandasDataFrameCleaner:
        """
        Normalizes the string in the specified column to the specified normalization form.

        Args:
            column (str): The name of the column.
            form (str, optional): The normalization form ('NFC', 'NFD', 'NFKC', 'NFKD'). Default is 'NFC'.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].str.normalize(form)

        return self

    def get_dummies(self, column: str, prefix: str = None) -> PandasDataFrameCleaner:
        """
        Converts categorical variable(s) in the specified column into dummy/indicator variables.

        Args:
            column (str): The name of the column to convert.
            prefix (str, optional): The prefix to append to the dummy variable names.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        dummies = pd.get_dummies(self.dataframe[column], prefix=prefix)
        self.dataframe = pd.concat([self.dataframe, dummies], axis=1)

        return self

    def normalize_data(
        self, columns: List[str], scaler: MinMaxScaler = MinMaxScaler()
    ) -> PandasDataFrameCleaner:
        """
        Normalizes the specified columns using the given scaler.

        Args:
            columns (List[str]): List of column names to normalize.
            scaler (MinMaxScaler, optional): The scaler to use for normalization. Default is MinMaxScaler.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[columns] = scaler.fit_transform(self.dataframe[columns])

        return self

    def trim_data(self, start_row: int, end_row: int) -> PandasDataFrameCleaner:
        """
        Trims the DataFrame to include rows only within the specified range.

        Args:
            start_row (int): The starting row index.
            end_row (int): The ending row index.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe = self.dataframe.iloc[start_row:end_row]

        return self

    def standardize_data(self, column: str) -> PandasDataFrameCleaner:
        """
        Standardizes the data in the specified column.

        Args:
            column (str): The name of the column to standardize.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = (
            self.dataframe[column] - self.dataframe[column].mean()
        ) / self.dataframe[column].std()

        return self

    def correct_errors(
        self, column: str, correction_map: Dict[Any, Any]
    ) -> PandasDataFrameCleaner:
        """
        Corrects errors in the specified column based on a mapping dictionary.

        Args:
            column (str): The name of the column in which to correct errors.
            correction_map (Dict[Any, Any]): A dictionary mapping erroneous values to their corrections.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = self.dataframe[column].replace(correction_map)

        return self

    def validate_data(
        self, validation_rules: Dict[str, Callable[[Any], bool]]
    ) -> PandasDataFrameCleaner:
        """
        Validates the data in the DataFrame based on specified rules.

        Args:
            validation_rules (Dict[str, Callable[[Any], bool]]): A dictionary where keys are column names and values
            are functions that return a boolean indicating whether the data in the column passes the validation.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.

        Raises:
            ValueError: If any data in a column fails the validation rule.
        """
        for column, rule in validation_rules.items():
            if not self.dataframe[column].apply(rule).all():
                raise ValueError(f"Data validation failed for column {column}")

        return self

    def balance_data(
        self,
        target_column: str,
        method: Literal["resample", "undersample", "smote"] = "resample",
        **kwargs: Any,
    ) -> PandasDataFrameCleaner:
        """
        Balances the data in the DataFrame based on the target column using specified methods.

        Args:
            target_column (str): The name of the target column.
            method (Literal["resample", "undersample", "smote"], optional): The method to use for balancing.
                Can be "resample", "undersample", or "smote". Default is "resample".
            **kwargs (Any): Additional keyword arguments for the balancing methods.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        if method == "resample":
            majority_class = self.dataframe[target_column].mode()[0]
            minority_class_df = self.dataframe[
                self.dataframe[target_column] != majority_class
            ]
            majority_class_df = self.dataframe[
                self.dataframe[target_column] == majority_class
            ]

            majority_class_resampled = resample(
                majority_class_df,
                n_samples=len(minority_class_df),
                random_state=kwargs.get("random_state", 0),
            )
            self.dataframe = pd.concat([minority_class_df, majority_class_resampled])
        elif method == "undersample":
            sampler = RandomUnderSampler(**kwargs)
            X = self.dataframe.drop(target_column, axis=1)
            y = self.dataframe[target_column]
            X_res, y_res = sampler.fit_resample(X, y)
            self.dataframe = pd.DataFrame(X_res, columns=X.columns)
            self.dataframe[target_column] = y_res
        elif method == "smote":
            sampler = SMOTE(**kwargs)
            X = self.dataframe.drop(target_column, axis=1)
            y = self.dataframe[target_column]
            X_res, y_res = sampler.fit_resample(X, y)
            self.dataframe = pd.DataFrame(X_res, columns=X.columns)
            self.dataframe[target_column] = y_res

        return self

    def bin_data(
        self,
        column: str,
        bins: Union[int, np.ndarray, List[float]],
        labels: Optional[List[str]] = None,
    ) -> PandasDataFrameCleaner:
        """
        Bins the data in the specified column into discrete intervals.

        Args:
            column (str): The name of the column to bin.
            bins (Union[int, np.ndarray, List[float]]): Defines the bin edges. Can be an integer number of bins,
                or a sequence of bin edges.
            labels (Optional[List[str]], optional): Labels for the returned bins. Must be the same length as the resulting bins.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        self.dataframe[column] = pd.cut(
            self.dataframe[column], bins=bins, labels=labels
        )

        return self

    def encode_categorical_data(
        self,
        column: str,
        method: Literal["onehot", "label"] = "label",
    ) -> PandasDataFrameCleaner:
        """
        Encodes categorical data in the specified column using the specified method.

        Args:
            column (str): The name of the column to encode.
            method (Literal["onehot", "label"], optional): The method of encoding, either 'onehot' or 'label'. Default is 'label'.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        methods = {
            "onehot": OneHotEncoderMethod,
            "label": LabelEncoderMethod,
        }

        encoder = methods[method].encoder

        if method == "onehot":
            encoded = encoder.fit_transform(self.dataframe[[column]])
            for i, category in enumerate(encoder.categories_[0]):
                self.dataframe[f"{column}_{category}"] = encoded[:, i].toarray()
        elif method == "label":
            self.dataframe[column] = encoder.fit_transform(self.dataframe[column])

        return self

    def handle_outliers(
        self,
        column: str,
        method: Literal[
            "clip",
            "remove",
            "replace",
            "log_transform",
            "std_dev",
            "winsorize",
            "z_score",
        ] = "clip",
        **kwargs: Any,
    ) -> PandasDataFrameCleaner:
        """
        Handles outliers in the specified column using the specified method.

        Args:
            column (str): The name of the column to handle outliers in.
            method (Literal[...], optional): The method for handling outliers. Options include 'clip', 'remove',
                'replace', 'log_transform', 'std_dev', 'winsorize', and 'z_score'. Default is 'clip'.
            **kwargs (Any): Additional keyword arguments specific to the outlier handling method chosen.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        if method == "clip":
            lower = kwargs.get("lower", self.dataframe[column].quantile(0.01))
            upper = kwargs.get("upper", self.dataframe[column].quantile(0.99))
            self.dataframe[column] = self.dataframe[column].clip(lower, upper)
        elif method == "remove":
            lower = kwargs.get("lower", self.dataframe[column].quantile(0.01))
            upper = kwargs.get("upper", self.dataframe[column].quantile(0.99))
            self.dataframe = self.dataframe[
                (self.dataframe[column] >= lower) & (self.dataframe[column] <= upper)
            ]
        elif method == "replace":
            lower = kwargs.get("lower", self.dataframe[column].quantile(0.01))
            upper = kwargs.get("upper", self.dataframe[column].quantile(0.99))
            replace_with = kwargs.get("replace_with", self.dataframe[column].median())
            condition = ~(
                (self.dataframe[column] >= lower) & (self.dataframe[column] <= upper)
            )
            self.dataframe.loc[condition, column] = replace_with
        elif method == "log_transform":
            offset = kwargs.get("offset", 1)
            self.dataframe[column] = np.log(self.dataframe[column] + offset)
        elif method == "std_dev":
            num_std_dev = kwargs.get("num_std_dev", 2)
            mean = self.dataframe[column].mean()
            std_dev = self.dataframe[column].std()
            lower_bound = mean - (num_std_dev * std_dev)
            upper_bound = mean + (num_std_dev * std_dev)
            if kwargs.get("remove", False):
                self.dataframe = self.dataframe[
                    (self.dataframe[column] >= lower_bound)
                    & (self.dataframe[column] <= upper_bound)
                ]
            else:
                replace_with = kwargs.get("replace_with", mean)
                condition = ~(
                    (self.dataframe[column] >= lower_bound)
                    & (self.dataframe[column] <= upper_bound)
                )
                self.dataframe.loc[condition, column] = replace_with
        elif method == "winsorize":
            limits = kwargs.get("limits", (0.01, 0.99))
            self.dataframe[column] = winsorize(self.dataframe[column], limits=limits)

        elif method == "z_score":
            threshold = kwargs.get("threshold", 3)
            zs = np.abs(zscore(self.dataframe[column]))
            self.dataframe = self.dataframe[zs < threshold]

        return self

    def detect_anomalies(
        self,
        column: str,
        method: Literal[
            "iqr",
            "z_score",
            "modified_z_score",
            "dbscan",
            "isolation_forest",
            "kmeans",
        ] = "iqr",
        threshold: float = 1.5,
        flag_column_format: str = "{column}_anomaly_flag",
        **kwargs: Any,
    ) -> PandasDataFrameCleaner:
        """
        Detects anomalies in the specified column using the chosen method.

        Args:
            column (str): The name of the column to detect anomalies in.
            method (Literal[...]): The method to use for anomaly detection. Options are 'iqr', 'z_score',
                'modified_z_score', 'dbscan', 'isolation_forest', and 'kmeans'.
            threshold (float, optional): The threshold value specific to the method used. Default is 1.5.
            flag_column_format (str, optional): Format for the name of the column that flags anomalies. Default is '{column}_anomaly_flag'.
            **kwargs (Any): Additional keyword arguments specific to the anomaly detection method chosen.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        flag_column = flag_column_format.format(column)
        if method == "iqr":
            Q1 = self.dataframe[column].quantile(0.25)
            Q3 = self.dataframe[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            self.dataframe[flag_column] = ~self.dataframe[column].between(
                lower_bound, upper_bound
            )
        elif method == "z_score":
            zs = np.abs(zscore(self.dataframe[column]))
            self.dataframe[flag_column] = zs > threshold
        elif method == "modified_z_score":
            median = self.dataframe[column].median()
            mad = median_abs_deviation(self.dataframe[column])
            modified_z_scores = 0.6745 * (self.dataframe[column] - median) / mad
            self.dataframe[flag_column] = np.abs(modified_z_scores) > threshold
        elif method == "dbscan":
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.dataframe[[column]])
            db = DBSCAN(
                eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5)
            ).fit(scaled_data)
            self.dataframe[flag_column] = db.labels_ == -1
        elif method == "isolation_forest":
            clf = IsolationForest(contamination=kwargs.get("contamination", "auto"))
            self.dataframe[flag_column] = (
                clf.fit_predict(self.dataframe[[column]]) == -1
            )
        elif method == "kmeans":
            kmeans = KMeans(n_clusters=kwargs.get("n_clusters", 3))
            self.dataframe["cluster"] = kmeans.fit_predict(self.dataframe[[column]])
            centers = kmeans.cluster_centers_
            distance = np.linalg.norm(
                self.dataframe[[column]] - centers[self.dataframe["cluster"]], axis=1
            )
            self.dataframe[flag_column] = distance > threshold

        return self

    def remove_low_variance(self, threshold: float = 0.0) -> PandasDataFrameCleaner:
        """
        Removes columns from the DataFrame that have variance below a specified threshold.

        Args:
            threshold (float, optional): The threshold for variance below which columns will be removed. Default is 0.0.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        selector = VarianceThreshold(threshold)
        self.dataframe = pd.DataFrame(
            selector.fit_transform(self.dataframe),
            columns=self.dataframe.columns[selector.get_support()],
        )

        return self

    def remove_high_correlation(self, threshold: float = 0.8) -> PandasDataFrameCleaner:
        """
        Removes columns from the DataFrame that have a high correlation with other columns.

        Args:
            threshold (float, optional): The correlation threshold above which columns will be removed. Default is 0.8.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        corr_matrix = self.dataframe.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        self.dataframe.drop(to_drop, axis=1, inplace=True)

        return self

    def remove_low_correlation(
        self, target_column: str, threshold: float = 0.2
    ) -> PandasDataFrameCleaner:
        """
        Removes columns from the DataFrame that have a low correlation with the target column.

        Args:
            target_column (str): The name of the target column to correlate with.
            threshold (float, optional): The correlation threshold below which columns will be removed. Default is 0.2.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        correlation_with_target = self.dataframe.corrwith(
            self.dataframe[target_column]
        ).abs()
        low_corr_columns = [
            column
            for column in correlation_with_target.index
            if correlation_with_target[column] < threshold
        ]
        self.dataframe.drop(low_corr_columns, axis=1, inplace=True)

        return self

    def select_features_model_importance(
        self, target_column: str, n_features: int = 10
    ) -> PandasDataFrameCleaner:
        """
        Selects a specified number of features based on their importance as determined by a RandomForestClassifier model.

        Args:
            target_column (str): The name of the target column for the model.
            n_features (int, optional): The number of features to select. Default is 10.

        Returns:
            PandasDataFrameCleaner: The modified DataFrame cleaner instance.
        """
        X = self.dataframe.drop(target_column, axis=1)
        y = self.dataframe[target_column]
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[-n_features:]
        self.dataframe = self.dataframe.iloc[:, indices]

        return self


if __name__ == "__main__":
    # Example usage:
    df = pd.DataFrame(...)  # Your pandas DataFrame
    cleaner = PandasDataFrameCleaner(df)
    cleaner.drop_missing_values()
    cleaner.fill_missing_values(0)
    cleaner.remove_duplicates()
    translation_table = str.maketrans("abc", "123")
    cleaner.string_translate("column_name", translation_table)
    cleaner.string_get_dummies("column_with_multiple_labels", separator="|")
    cleaner.string_normalize("column_name", form="NFC")
    cleaner.string_encode("column_name", encoding="utf-8")
    cleaner.string_decode("column_name", encoding="utf-8")
    cleaner.string_lower("your_column")
    cleaner.string_upper("your_column")
    cleaner.string_remove_prefix("column_name", "pre_")
    cleaner.string_remove_suffix("column_name", "_suf")
    cleaner.string_pad("column_name", width=10, side="left", fillchar="_")
    cleaner.string_center("column_name", width=10, fillchar="_")
    cleaner.string_ljust("column_name", width=10, fillchar="_")
    cleaner.string_rjust("column_name", width=10, fillchar="_")
    cleaner.string_zfill("column_name", width=10)
    cleaner.string_slice("column_name", start=2, stop=5)
    cleaner.string_slice_replace("column_name", start=2, stop=5, repl="new")
    cleaner.string_extract("column_name", regex="(pattern)", expand=True)
    cleaner.string_extract_all("column_name", regex="(pattern)")
    cleaner.string_remove_punctuation("text_column")
    cleaner.string_remove_whitespace("text_column")
    cleaner.string_remove_extra_spaces("text_column")
    cleaner.string_replace_specific_words(
        "text_column", {"old_word": "new_word", "another_old_word": "another_new_word"}
    )
    cleaner.normalize_data(["column1", "column2"])
    cleaner.trim_data(0, 10)
    cleaner.handle_outliers("column_name", method="clip", lower=10, upper=90)
    cleaner.handle_outliers("column_name", method="remove", lower=10, upper=90)
    cleaner.handle_outliers(
        "column_name", method="replace", lower=10, upper=90, replace_with=50
    )
    cleaner.handle_outliers("column_name", method="log_transform", offset=1)
    cleaner.handle_outliers("column_name", method="std_dev", num_std_dev=3, remove=True)
    cleaner.handle_outliers(
        "column_name", method="std_dev", num_std_dev=3, replace_with=None
    )
    cleaner.standardize_data("column_name")
    cleaner.encode_categorical_data("category_column", method="label")
    cleaner.validate_data({"column_name": lambda x: x > 0})
    cleaner.feature_engineering(
        "new_feature", lambda row, arg1: row["column1"] * arg1, 2
    )
    cleaner.bin_data("column_to_bin", bins=3, labels=["Low", "Medium", "High"])
    cleaner.correct_errors("column_name", {"error_value": "correct_value"})
    cleaner.get_dummies("column_name", prefix="dummy")
    cleaner.balance_data("target_column", method="resample")
    cleaner.balance_data("target_column", method="undersample")
    cleaner.balance_data("target_column", method="smote")
    cleaner.detect_anomalies(
        "column_name",
        method="iqr",
        threshold=1.5,
        flag_column_format="{column}_outlier_flag",
    )
    cleaner.detect_anomalies(
        "column_name",
        method="modified_z_score",
        threshold=3.5,
        flag_column_format="{column}_mzscore_anomaly",
    )
    cleaner.detect_anomalies(
        "column_name",
        method="dbscan",
        eps=0.5,
        min_samples=5,
        flag_column_format="{column}_dbscan_anomaly",
    )
    cleaner.detect_anomalies(
        "column_name",
        method="isolation_forest",
        contamination=0.02,
        flag_column_format="{column}_iso_forest_anomaly",
    )
    cleaner.detect_anomalies(
        "column_name",
        method="kmeans",
        n_clusters=2,
        threshold=2,
        flag_column_format="{column}_kmeans_anomaly",
    )
    cleaner.remove_low_variance(threshold=0.01)
    cleaner.remove_high_correlation(threshold=0.85)
    cleaner.remove_low_correlation("target_column", threshold=0.2)
    cleaner.select_features_model_importance("target_column", n_features=5)
