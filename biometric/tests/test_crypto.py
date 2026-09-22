"""Round-trip tests for crypto.py — no key means plaintext passthrough."""

from django.test import SimpleTestCase

from biometric import crypto


class TestVectorRoundTrip(SimpleTestCase):

    def test_no_key_returns_plaintext(self):
        vector = [0.1, 0.2, 0.3]
        encrypted = crypto.encrypt_vector(vector, None)
        self.assertEqual(encrypted, vector)
        self.assertEqual(crypto.decrypt_vector(encrypted, None), vector)

    def test_with_key_round_trips(self):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        vector = [0.1, -0.2, 0.375]
        encrypted = crypto.encrypt_vector(vector, key)

        self.assertNotEqual(encrypted, vector)
        self.assertIsInstance(encrypted, str)
        self.assertEqual(crypto.decrypt_vector(encrypted, key), vector)

    def test_none_vector_stays_none(self):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        self.assertIsNone(crypto.encrypt_vector(None, key))
        self.assertIsNone(crypto.decrypt_vector(None, key))

    def test_decrypt_with_wrong_key_falls_back_to_raw_value(self):
        from cryptography.fernet import Fernet

        key_a = Fernet.generate_key()
        key_b = Fernet.generate_key()
        encrypted = crypto.encrypt_vector([1.0], key_a)
        # InvalidToken is swallowed; the raw (undecryptable) value is returned.
        self.assertEqual(crypto.decrypt_vector(encrypted, key_b), encrypted)


class TestBytesRoundTrip(SimpleTestCase):

    def test_no_key_returns_plaintext_bytes(self):
        data = b"\x00\x01template-bytes"
        encrypted = crypto.encrypt_bytes(data, None)
        self.assertEqual(encrypted, data)
        self.assertEqual(crypto.decrypt_bytes(encrypted, None), data)

    def test_with_key_round_trips(self):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        data = b"vendor-template-bytes"
        encrypted = crypto.encrypt_bytes(data, key)

        self.assertNotEqual(encrypted, data)
        self.assertEqual(crypto.decrypt_bytes(encrypted, key), data)

    def test_none_bytes_stays_none(self):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        self.assertIsNone(crypto.encrypt_bytes(None, key))
        self.assertIsNone(crypto.decrypt_bytes(None, key))


class TestWarnIfUnencrypted(SimpleTestCase):

    def test_warns_once_when_key_is_none(self):
        crypto._warned_no_key = False
        with self.assertLogs("biometric.crypto", level="WARNING") as ctx:
            crypto.warn_if_unencrypted(None)
        self.assertTrue(any("plaintext" in msg for msg in ctx.output))
        self.assertTrue(crypto._warned_no_key)
        crypto._warned_no_key = False

    def test_no_warning_when_key_is_set(self):
        crypto._warned_no_key = False
        with self.assertRaises(AssertionError):
            with self.assertLogs("biometric.crypto", level="WARNING"):
                crypto.warn_if_unencrypted("some-key")
