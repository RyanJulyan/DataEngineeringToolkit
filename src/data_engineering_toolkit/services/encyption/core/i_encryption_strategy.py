from abc import ABC, abstractmethod


class IEncryptionStrategy(ABC):
    """
    Abstract base class for encryption strategies.

    This class defines the interface that all encryption strategy implementations must follow.
    Concrete subclasses must implement the `encrypt` and `decrypt` methods.
    """

    @abstractmethod
    def encrypt(self, message: str) -> str:
        """
        Encrypts a given message.

        Args:
            message (str): The message to be encrypted.

        Returns:
            str: The encrypted message.
        """
        pass

    @abstractmethod
    def decrypt(self, encoded_message: str) -> str:
        """
        Decrypts a given encoded message.

        Args:
            encoded_message (str): The message to be decrypted.

        Returns:
            str: The decrypted message.
        """
        pass
