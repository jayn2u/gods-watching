"""Encryption boundary for RTSP source credentials."""

from dataclasses import dataclass
from typing import final, override

from cryptography.fernet import Fernet, InvalidToken


@final
class CredentialDecryptionError(Exception):
    """Report ciphertext that the configured key cannot authenticate."""

    @override
    def __str__(self) -> str:
        """Return a sanitized authentication failure."""
        return "stored camera source could not be decrypted"


@dataclass(frozen=True, slots=True)
class CredentialCipher:
    """Encrypt and authenticate camera source URLs at the repository boundary."""

    key: bytes

    def encrypt(self, source_url: str) -> bytes:
        """Encrypt and authenticate a source URL."""
        return Fernet(self.key).encrypt(source_url.encode())

    def decrypt(self, ciphertext: bytes) -> str:
        """Decrypt authenticated storage bytes without logging secrets."""
        try:
            return Fernet(self.key).decrypt(ciphertext).decode()
        except InvalidToken as error:
            raise CredentialDecryptionError from error
