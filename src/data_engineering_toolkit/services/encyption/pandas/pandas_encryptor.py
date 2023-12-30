from __future__ import annotations
from dataclasses import dataclass
from typing import Union

import pandas as pd

from data_engineering_toolkit.services.encyption.core.encryption_strategies import (
    AnonymizationEncryptionStrategy,
    DiffieHellmanEncryptionStrategy,
    PseudoAnonymizationEncryptionStrategy,
    ScrambleEncryptionStrategy,
    SubstitutionEncryptionStrategy,
)


@dataclass
class PandasDataFrameEncryptor:
    """
    A class for encrypting and decrypting columns of a pandas DataFrame using various encryption strategies.

    Attributes:
        strategy: The encryption strategy to be used.
        data: The pandas DataFrame to be encrypted/decrypted.

    Methods:
        encrypt_column: Encrypts a specified column in the DataFrame.
        decrypt_column: Decrypts a specified column in the DataFrame.
    """

    def __init__(
        self,
        strategy: Union[
            AnonymizationEncryptionStrategy,
            DiffieHellmanEncryptionStrategy,
            PseudoAnonymizationEncryptionStrategy,
            ScrambleEncryptionStrategy,
            SubstitutionEncryptionStrategy,
        ],
        data: pd.DataFrame,
    ) -> None:
        """
        Initializes the PandasDataFrameEncryptor with a specified encryption strategy and DataFrame.

        Args:
            strategy: An instance of one of the encryption strategy classes.
            data: The pandas DataFrame that will be subject to encryption/decryption.
        """
        self.strategy: Union[
            AnonymizationEncryptionStrategy,
            DiffieHellmanEncryptionStrategy,
            PseudoAnonymizationEncryptionStrategy,
            ScrambleEncryptionStrategy,
            SubstitutionEncryptionStrategy,
        ] = strategy  # Instance of a class that inherits from EncryptionStrategy
        self.data: pd.DataFrame = data  # Pandas DataFrame

    def encrypt_column(self, column: str) -> PandasDataFrameEncryptor:
        """
        Encrypts a specified column in the DataFrame.

        Args:
            column (str): The name of the column to encrypt.

        Returns:
            PandasDataFrameEncryptor: The instance itself, allowing for method chaining.
        """
        self.data[column] = self.data[column].apply(
            lambda x: self.strategy.encryption_strategy.encrypt(x)
        )

        return self

    def decrypt_column(self, column: str) -> PandasDataFrameEncryptor:
        """
        Decrypts a specified column in the DataFrame.

        Args:
            column (str): The name of the column to decrypt.

        Returns:
            PandasDataFrameEncryptor: The instance itself, allowing for method chaining.
        """
        self.data[column] = self.data[column].apply(
            lambda x: self.strategy.encryption_strategy.decrypt(x)
        )

        return self
