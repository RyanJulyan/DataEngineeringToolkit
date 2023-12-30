from binascii import hexlify
from hashlib import sha256
from os import urandom

from data_engineering_toolkit.services.encyption.core.i_encryption_strategy import (
    IEncryptionStrategy,
)
from data_engineering_toolkit.services.encyption.core.primes import primes


class DiffieHellmanEncryption(IEncryptionStrategy):
    """
    Implementation of the Diffie-Hellman encryption strategy.

    This class provides methods for generating private/public keys, shared keys,
    and encrypting/decrypting messages using the Diffie-Hellman key exchange protocol.

    Attributes:
        prime (int): The prime number used for key generation.
        generator (int): The generator value used for key generation.
        __private_key (int): The private key generated for this instance.
    """

    # Current minimum recommendation is 2048 bit (encoding_bit 14)
    def __init__(self, encoding_bit: int = 14) -> None:
        """
        Initializes a DiffieHellmanEncryption instance with a specified encoding bit.

        Args:
            encoding_bit (int): The encoding bit determining the prime and generator values.

        Raises:
            ValueError: If the encoding bit is not supported.
        """
        if encoding_bit not in primes:
            raise ValueError("Unsupported encoding_bit")
        self.prime = primes[encoding_bit]["prime"]
        self.generator = primes[encoding_bit]["generator"]

        self.__private_key = int(hexlify(urandom(32)), base=16)

    def get_private_key(self) -> str:
        """
        Retrieves the hex representation of the private key.

        Returns:
            str: The private key in hexadecimal format.
        """
        return hex(self.__private_key)[2:]

    def generate_public_key(self) -> str:
        """
        Generates a public key based on the private key and preset values.

        Returns:
            str: The public key in hexadecimal format.
        """
        public_key = pow(self.generator, self.__private_key, self.prime)
        return hex(public_key)[2:]

    def is_valid_public_key(self, key: int) -> bool:
        """
        Validates a given public key based on NIST SP800-56 standards.

        Args:
            key (int): The public key to validate.

        Returns:
            bool: True if the key is valid, False otherwise.
        """
        # check if the other public key is valid based on NIST SP800-56
        if 2 <= key and key <= self.prime - 2:
            if pow(key, (self.prime - 1) // 2, self.prime) == 1:
                return True
        return False

    def generate_shared_key(self, other_key_str: str) -> str:
        """
        Generates a shared key using the provided public key and the private key of this instance.

        Args:
            other_key_str (str): The public key of the other party in hexadecimal format.

        Returns:
            str: The generated shared key in hexadecimal format.

        Raises:
            ValueError: If the provided public key is invalid.
        """
        other_key = int(other_key_str, base=16)
        if not self.is_valid_public_key(other_key):
            raise ValueError("Invalid public key")
        shared_key = pow(other_key, self.__private_key, self.prime)
        return sha256(str(shared_key).encode()).hexdigest()

    @staticmethod
    def is_valid_public_key_static(
        local_private_key_str: str, remote_public_key_str: str, prime: int
    ) -> bool:
        """
        Static method to validate a public key based on NIST SP800-56 standards.

        Args:
            local_private_key_str (str): The private key of the local party in hexadecimal format.
            remote_public_key_str (str): The public key of the remote party in hexadecimal format.
            prime (int): The prime number used for key generation.

        Returns:
            bool: True if the remote public key is valid, False otherwise.
        """
        # check if the other public key is valid based on NIST SP800-56
        if 2 <= remote_public_key_str and remote_public_key_str <= prime - 2:
            if pow(remote_public_key_str, (prime - 1) // 2, prime) == 1:
                return True
        return False

    @staticmethod
    def generate_shared_key_static(
        local_private_key_str: str, remote_public_key_str: str, encoding_bit: int = 14
    ) -> str:
        """
        Static method to generate a shared key using the Diffie-Hellman key exchange protocol.

        Args:
            local_private_key_str (str): The private key of the local party in hexadecimal format.
            remote_public_key_str (str): The public key of the remote party in hexadecimal format.
            encoding_bit (int): The encoding bit determining the prime and generator values.

        Returns:
            str: The generated shared key in hexadecimal format.

        Raises:
            ValueError: If the remote public key is invalid.
        """
        local_private_key = int(local_private_key_str, base=16)
        remote_public_key = int(remote_public_key_str, base=16)
        prime = primes[encoding_bit]["prime"]
        if not DiffieHellmanEncryption.is_valid_public_key_static(
            local_private_key, remote_public_key, prime
        ):
            raise ValueError("Invalid public key")
        shared_key = pow(remote_public_key, local_private_key, prime)
        return sha256(str(shared_key).encode()).hexdigest()

    @staticmethod
    def xor_encrypt_decrypt(message: str, key_string: str):
        """
        Encrypts or decrypts a message using the XOR operation with a given key.

        Args:
            message (str): The message to encrypt or decrypt.
            key_string (str): The key used for the XOR operation.

        Returns:
            str: The resulting encrypted or decrypted message.
        """
        key = list(key_string)
        output = []
        for i in range(len(message)):
            char_code = ord(message[i]) ^ ord(key[i % len(key)][0])
            output.append(chr(char_code))
        return "".join(output)

    @staticmethod
    def encrypt(message: str, key: str):
        """
        Encrypts a message using the Diffie-Hellman encryption method.

        Args:
            message (str): The message to be encrypted.
            key (str): The encryption key.

        Returns:
            str: The encrypted message.
        """
        return DiffieHellmanEncryption.xor_encrypt_decrypt(message, key)

    @staticmethod
    def decrypt(encrypted_message: str, key: str):
        """
        Decrypts a message encrypted with the Diffie-Hellman encryption method.

        Args:
            encrypted_message (str): The encrypted message.
            key (str): The decryption key.

        Returns:
            str: The decrypted message.
        """
        return DiffieHellmanEncryption.xor_encrypt_decrypt(encrypted_message, key)
