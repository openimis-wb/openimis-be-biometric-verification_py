"""
Unit tests for services.py

Covers:
- _decode_frame: strips data URI prefix, decodes base64
- _fetch_insuree_photo: correct path construction, missing photo errors
- verify_face: fast path (stored embedding), slow path (image comparison), error handling
- compute_insuree_embedding: creates record, invalidates old embedding
"""
import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from django.test import SimpleTestCase

from biometric_verification.providers.base import VerificationResult
from biometric_verification.services import BiometricService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(data: bytes = b"fake-image") -> str:
    return base64.b64encode(data).decode()


def _b64_data_uri(data: bytes = b"fake-image") -> str:
    return f"data:image/jpeg;base64,{_b64(data)}"


def _make_verified_result(**kwargs):
    return VerificationResult(
        verified=True, confidence=92.0, distance=0.08, provider="stub", **kwargs
    )


# ---------------------------------------------------------------------------
# _decode_frame
# ---------------------------------------------------------------------------

class TestDecodeFrame(SimpleTestCase):

    def test_plain_base64(self):
        raw = b"hello world"
        result = BiometricService._decode_frame(_b64(raw))
        self.assertEqual(result, raw)

    def test_data_uri_prefix_stripped(self):
        raw = b"hello world"
        result = BiometricService._decode_frame(_b64_data_uri(raw))
        self.assertEqual(result, raw)

    def test_jpeg_prefix_stripped(self):
        raw = b"\xff\xd8\xff"  # JPEG magic bytes
        encoded = f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"
        result = BiometricService._decode_frame(encoded)
        self.assertEqual(result, raw)


# ---------------------------------------------------------------------------
# _fetch_insuree_photo
# ---------------------------------------------------------------------------

class TestFetchInsureePhoto(SimpleTestCase):

    def _make_insuree(self, folder="2024/01", filename="photo.jpg"):
        photo = MagicMock()
        photo.folder = folder
        photo.filename = filename
        insuree = MagicMock()
        insuree.photo = photo
        insuree.uuid = "test-uuid"
        return insuree

    @patch("insuree.apps.InsureeConfig")
    def test_returns_file_bytes(self, mock_cfg):
        # Arrange
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            photo_dir = tmp_path / "2024" / "01"
            photo_dir.mkdir(parents=True)
            photo_file = photo_dir / "photo.jpg"
            photo_file.write_bytes(b"JPEG_DATA")

            mock_cfg.insuree_photos_root_path = str(tmp_path)
            insuree = self._make_insuree()

            # Act
            result = BiometricService._fetch_insuree_photo(insuree)

        self.assertEqual(result, b"JPEG_DATA")

    @patch("insuree.apps.InsureeConfig")
    def test_no_photo_raises_value_error(self, mock_cfg):
        mock_cfg.insuree_photos_root_path = "/some/root"
        insuree = MagicMock()
        insuree.photo = None
        insuree.uuid = "test-uuid"

        with self.assertRaises(ValueError, msg="no reference photo"):
            BiometricService._fetch_insuree_photo(insuree)

    @patch("insuree.apps.InsureeConfig")
    def test_missing_root_path_raises_value_error(self, mock_cfg):
        mock_cfg.insuree_photos_root_path = None
        insuree = self._make_insuree()

        with self.assertRaises(ValueError, msg="not configured"):
            BiometricService._fetch_insuree_photo(insuree)

    @patch("insuree.apps.InsureeConfig")
    def test_file_not_found_raises(self, mock_cfg):
        mock_cfg.insuree_photos_root_path = "/nonexistent/root"
        insuree = self._make_insuree()

        with self.assertRaises(FileNotFoundError):
            BiometricService._fetch_insuree_photo(insuree)


# ---------------------------------------------------------------------------
# verify_face
# ---------------------------------------------------------------------------

class TestVerifyFace(SimpleTestCase):

    def _mock_provider(self, result=None):
        provider = MagicMock()
        provider.verify.return_value = result or _make_verified_result()
        provider.verify_from_embedding.return_value = result or _make_verified_result()
        return provider

    def _mock_insuree(self, embedding=None):
        embedding_obj = MagicMock()
        embedding_obj.embedding = embedding or [0.1, 0.2, 0.3]
        embedding_obj.is_active = True

        insuree = MagicMock()
        insuree.uuid = "insuree-uuid"
        insuree.biometric_embedding = embedding_obj
        return insuree

    @patch("biometric_verification.services.BiometricService._fetch_insuree_photo")
    @patch("biometric_verification.registry.ProviderRegistry")
    @patch("insuree.models.Insuree")
    def test_fast_path_uses_stored_embedding(
        self, mock_insuree_cls, mock_registry, mock_fetch
    ):
        from biometric_verification.apps import BiometricVerificationConfig

        provider = self._mock_provider()
        mock_registry.get_active_provider.return_value = provider
        insuree = self._mock_insuree()
        # Stored config must match current config (provider_config patched to {}
        # below) so verify_face takes the fast path instead of recomputing.
        insuree.biometric_embedding.metadata = {
            "model_name": provider.model_name,
            "provider": provider.provider_name,
        }
        mock_insuree_cls.objects.get.return_value = insuree

        with patch.object(BiometricVerificationConfig, "store_embeddings", True), \
                patch.object(BiometricVerificationConfig, "provider_config", {}):
            result = BiometricService.verify_face(
                insuree_uuid="insuree-uuid",
                frame_b64=_b64(),
                user=MagicMock(),
            )

        provider.verify_from_embedding.assert_called_once()
        provider.verify.assert_not_called()
        mock_fetch.assert_not_called()
        self.assertTrue(result.verified)

    @patch("biometric_verification.apps.BiometricVerificationConfig")
    @patch("biometric_verification.services.BiometricService._fetch_insuree_photo")
    @patch("biometric_verification.registry.ProviderRegistry")
    @patch("insuree.models.Insuree")
    def test_slow_path_used_when_no_embedding(
        self, mock_insuree_cls, mock_registry, mock_fetch, mock_cfg
    ):
        mock_cfg.store_embeddings = True
        provider = self._mock_provider()
        mock_registry.get_active_provider.return_value = provider

        # Insuree with no stored embedding
        insuree = MagicMock()
        insuree.uuid = "insuree-uuid"
        type(insuree).biometric_embedding = PropertyMock(
            side_effect=Exception("RelatedObjectDoesNotExist")
        )
        mock_insuree_cls.objects.get.return_value = insuree
        mock_fetch.return_value = b"PHOTO_DATA"

        result = BiometricService.verify_face(
            insuree_uuid="insuree-uuid",
            frame_b64=_b64(),
            user=MagicMock(),
        )

        provider.verify.assert_called_once()
        self.assertTrue(result.verified)

    @patch("biometric_verification.apps.BiometricVerificationConfig")
    @patch("biometric_verification.registry.ProviderRegistry")
    @patch("insuree.models.Insuree")
    def test_insuree_not_found_returns_error(
        self, mock_insuree_cls, mock_registry, mock_cfg
    ):
        mock_cfg.store_embeddings = False
        mock_insuree_cls.objects.get.side_effect = Exception("DoesNotExist")

        result = BiometricService.verify_face(
            insuree_uuid="bad-uuid",
            frame_b64=_b64(),
            user=MagicMock(),
        )

        self.assertFalse(result.verified)
        self.assertIsNotNone(result.error)


# ---------------------------------------------------------------------------
# compute_insuree_embedding
# ---------------------------------------------------------------------------

class TestComputeInsureeEmbedding(SimpleTestCase):

    @patch("django.utils.timezone")
    @patch("biometric_verification.models.BiometricEmbedding")
    @patch("biometric_verification.services.BiometricService._fetch_insuree_photo")
    @patch("biometric_verification.registry.ProviderRegistry")
    @patch("insuree.models.Insuree")
    def test_creates_embedding_and_invalidates_old(
        self, mock_insuree_cls, mock_registry, mock_fetch, mock_embedding_cls, mock_tz
    ):
        provider = MagicMock()
        provider.get_embedding.return_value = [0.1] * 512
        provider.model_name = "ArcFace"
        provider.provider_name = "deepface"
        mock_registry.get_active_provider.return_value = provider

        insuree = MagicMock()
        insuree.uuid = "insuree-uuid"
        mock_insuree_cls.objects.get.return_value = insuree
        mock_fetch.return_value = b"PHOTO"

        result = BiometricService.compute_insuree_embedding(
            insuree_uuid="insuree-uuid",
            user=MagicMock(),
        )

        # Old embedding invalidated
        mock_embedding_cls.objects.filter.assert_called_once_with(
            insuree=insuree, validity_to__isnull=True
        )
        mock_embedding_cls.objects.filter.return_value.update.assert_called_once()

        # New embedding created
        mock_embedding_cls.objects.create.assert_called_once()
        create_kwargs = mock_embedding_cls.objects.create.call_args.kwargs
        self.assertEqual(create_kwargs["model_name"], "ArcFace")
        self.assertEqual(create_kwargs["provider"], "deepface")
        self.assertEqual(len(create_kwargs["embedding"]), 512)

        self.assertTrue(result.success)
        self.assertEqual(result.model, "ArcFace")
        self.assertEqual(result.provider, "deepface")

    @patch("biometric_verification.registry.ProviderRegistry")
    @patch("insuree.models.Insuree")
    def test_returns_error_on_exception(self, mock_insuree_cls, mock_registry):
        mock_insuree_cls.objects.get.side_effect = Exception("DB error")

        result = BiometricService.compute_insuree_embedding(
            insuree_uuid="bad-uuid",
            user=MagicMock(),
        )

        self.assertFalse(result.success)
        self.assertIn("DB error", result.error)
