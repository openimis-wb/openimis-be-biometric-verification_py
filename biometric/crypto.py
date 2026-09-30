"""
Fernet encryption for template/vector at-rest storage.

Vectors and templates are plaintext only inside the service layer; models
store either plaintext or Fernet ciphertext depending on whether
BIOMETRIC["TEMPLATE_KEY"] is set, and BiometricTemplate.encrypted records
which. This module never reads config itself — callers pass the key
explicitly (row_key() maps the row's flag to it) so it stays testable
without touching BiometricConfig.
"""

import json
import logging

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_warned_no_key = False


class TemplateKeyError(ValueError):
    """
    A row marked encrypted does not decrypt with the configured TEMPLATE_KEY,
    or no key is configured. The message never quotes the stored value.
    """

    def __init__(self, message="A template marked encrypted does not decrypt with BIOMETRIC['TEMPLATE_KEY']."):
        # graphql-core copies .extensions onto the GraphQL error it reports.
        self.extensions = {"code": "BIOMETRIC_TEMPLATE_KEY"}
        super().__init__(message)


def row_key(encrypted, key):
    """The key to decrypt a row with: key when the row is marked encrypted, None for a plaintext row."""
    if not encrypted:
        return None
    if key is None:
        raise TemplateKeyError("A template is marked encrypted but BIOMETRIC['TEMPLATE_KEY'] is not set.")
    return key


def warn_if_unencrypted(key):
    """Log once at startup when TEMPLATE_KEY is unset (plaintext at rest)."""
    global _warned_no_key
    if key is None and not _warned_no_key:
        logger.warning(
            "BIOMETRIC['TEMPLATE_KEY'] is not set — biometric "
            "vectors and templates are stored in plaintext."
        )
        _warned_no_key = True


def _fernet(key) -> Fernet:
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_vector(vector, key):
    """Return the vector unchanged (key=None) or as Fernet-encrypted JSON text."""
    if key is None or vector is None:
        return vector
    return _fernet(key).encrypt(json.dumps(vector).encode()).decode()


def decrypt_vector(value, key):
    """
    Inverse of encrypt_vector. key=None returns the value unchanged (a
    plaintext row); a value the key does not decrypt raises TemplateKeyError.
    """
    if value is None:
        return None
    if key is None:
        return value
    try:
        payload = _fernet(key).decrypt(
            value.encode() if isinstance(value, str) else value
        )
    except (InvalidToken, TypeError):
        raise TemplateKeyError() from None
    return json.loads(payload)


def encrypt_bytes(data, key):
    if key is None or data is None:
        return data
    return _fernet(key).encrypt(bytes(data))


def decrypt_bytes(data, key):
    """Inverse of encrypt_bytes, with the same key=None and TemplateKeyError rules as decrypt_vector."""
    if data is None:
        return None
    if key is None:
        return bytes(data)
    try:
        return _fernet(key).decrypt(bytes(data))
    except (InvalidToken, TypeError):
        raise TemplateKeyError() from None
