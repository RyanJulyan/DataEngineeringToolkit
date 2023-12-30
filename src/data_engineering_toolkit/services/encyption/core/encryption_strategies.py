from dataclasses import dataclass
import os

from data_engineering_toolkit.services.encyption.core.i_encryption_strategy import (
    IEncryptionStrategy,
)
from data_engineering_toolkit.services.encyption.core.anonymization_encryption import (
    AnonymizationEncryption,
)
from data_engineering_toolkit.services.encyption.core.diffie_hellman_encryption import (
    DiffieHellmanEncryption,
)
from data_engineering_toolkit.services.encyption.core.pseudoanonymization_encryption import (
    PseudoAnonymizationEncryption,
)
from data_engineering_toolkit.services.encyption.core.scramble_encryption import (
    ScrambleEncryption,
)
from data_engineering_toolkit.services.encyption.core.substitution_encryption import (
    SubstitutionEncryption,
)


@dataclass
class AnonymizationEncryptionStrategy:
    """
    Strategy class for Anonymization Encryption.

    Attributes:
        encryption_strategy (IEncryptionStrategy): The encryption strategy object.
    """

    def __init__(
        self,
        salt: str,
        salt_multiplier: int = 5,
        app_iters: int = 1_750_000,
        hashfunc: str = "sha512",
        message_encoding: str = "utf-8",
    ) -> None:
        self.encryption_strategy: IEncryptionStrategy = AnonymizationEncryption(
            salt=salt,
            salt_multiplier=salt_multiplier,
            app_iters=app_iters,
            hashfunc=hashfunc,
            message_encoding=message_encoding,
        )


@dataclass
class DiffieHellmanEncryptionStrategy:
    """
    Strategy class for Diffie-Hellman Encryption.

    Attributes:
        encryption_strategy (IEncryptionStrategy): The encryption strategy object.
    """

    def __init__(
        self,
        encoding_bit: int = 14,
    ) -> None:
        self.encryption_strategy: IEncryptionStrategy = DiffieHellmanEncryption(
            encoding_bit=encoding_bit,
        )


@dataclass
class PseudoAnonymizationEncryptionStrategy:
    """
    Strategy class for Pseudo Anonymization Encryption.

    Attributes:
        encryption_strategy (IEncryptionStrategy): The encryption strategy object.
    """

    def __init__(
        self,
        salt: str,
        salt_multiplier: int = 5,
        message_encoding: str = "utf-8",
    ) -> None:
        self.encryption_strategy: IEncryptionStrategy = PseudoAnonymizationEncryption(
            salt=salt,
            salt_multiplier=salt_multiplier,
            message_encoding=message_encoding,
        )


@dataclass
class ScrambleEncryptionStrategy:
    """
    Strategy class for Scramble Encryption.

    Attributes:
        encryption_strategy (IEncryptionStrategy): The encryption strategy object.
    """

    def __init__(
        self,
        salt: str = "XX",
        salt_multiplier: int = 1,
        shift: int = 7,
    ) -> None:
        self.encryption_strategy: IEncryptionStrategy = ScrambleEncryption(
            salt=salt,
            salt_multiplier=salt_multiplier,
            shift=shift,
        )


@dataclass
class SubstitutionEncryptionStrategy:
    """
    Strategy class for Substitution Encryption.

    Attributes:
        strategy (IEncryptionStrategy): The encryption strategy object.
    """

    def __init__(
        self,
        current_key: str,
        substitution_lookup: dict,
        words_file_path: str = os.path.join(os.path.dirname(__file__), "words.txt"),
        update_substitution_loopup_kwags: dict = {
            "max_word_len": None,
            "min_word_len": None,
            "number_of_shuffles": 2,
        },
    ) -> None:
        self.strategy: IEncryptionStrategy = SubstitutionEncryption(
            current_key=current_key,
            substitution_lookup=substitution_lookup,
            words_file_path=words_file_path,
            update_substitution_loopup_kwags=update_substitution_loopup_kwags,
        )
