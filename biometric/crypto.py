"""
Fernet encryption for template/vector at-rest storage.

Vectors and templates are plaintext only inside the service layer; models
store either plaintext or Fernet ciphertext depending on whether
BIOMETRIC["TEMPLATE_KEY"] is set. This module never reads config itself —
callers pass the key explicitly so it stays testable without touching
BiometricConfig.
"""

import json
import logging

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_warned_no_key = False


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
    """Inverse of encrypt_vector — returns None/None and a plain list unchanged."""
    if value is None:
        return None
    if key is None:
        return value
    try:
        payload = _fernet(key).decrypt(
            value.encode() if isinstance(value, str) else value
        )
        return json.loads(payload)
    except InvalidToken:
        # Not actually encrypted (e.g. row written before TEMPLATE_KEY was set).
        return value


def encrypt_bytes(data, key):
    if key is None or data is None:
        return data
    return _fernet(key).encrypt(bytes(data))


def decrypt_bytes(data, key):
    if data is None:
        return None
    if key is None:
        return bytes(data)
    try:
        return _fernet(key).decrypt(bytes(data))
    except InvalidToken:
        return bytes(data)
