"""
BiometricVectorIndex sync (docs/wb-biometric-dedup-seam.md §6.2): create,
supersede, delete, consolidate, and the biometric_vector_reindex command.
"""

from django.core.management import call_command
from django.test import TestCase

from biometric.models import BiometricTemplate
from biometric.services import consolidate, enrol
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase
from biometric_pgvector.models import BiometricVectorIndex


class TestSync(_MultimodalServiceTestCase):

    def test_create_active_embedding_row_gets_side_row(self):
        template = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")

        side = BiometricVectorIndex.objects.get(template_id=template.id)
        self.assertEqual(side.modality, "face")
        self.assertEqual(side.provider, "fake_embedding")
        self.assertEqual(side.dim, len(side.embedding))

    def test_supersede_drops_the_stale_side_row(self):
        first = enrol(SUBJECT_MODEL, "s1", "face", b"photo-1", actor="tester")
        second = enrol(SUBJECT_MODEL, "s1", "face", b"photo-2", actor="tester")

        self.assertFalse(BiometricVectorIndex.objects.filter(template_id=first.id).exists())
        self.assertTrue(BiometricVectorIndex.objects.filter(template_id=second.id).exists())

    def test_delete_drops_the_side_row(self):
        template = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")
        self.assertTrue(BiometricVectorIndex.objects.filter(template_id=template.id).exists())

        BiometricTemplate.objects.filter(id=template.id).delete()

        self.assertFalse(BiometricVectorIndex.objects.filter(template_id=template.id).exists())

    def test_template_kind_row_never_gets_a_side_row(self):
        enrol(SUBJECT_MODEL, "s1", "fingerprint", b"template-bytes", actor="tester")
        self.assertEqual(BiometricVectorIndex.objects.count(), 0)

    def test_consolidate_collision_drops_retired_side_row(self):
        kept = enrol(SUBJECT_MODEL, "kept", "face", b"kept-photo", actor="tester")
        retired = enrol(SUBJECT_MODEL, "retired", "face", b"retired-photo", actor="tester")

        consolidate(SUBJECT_MODEL, "kept", "retired", actor="tester")

        self.assertTrue(BiometricVectorIndex.objects.filter(template_id=kept.id).exists())
        self.assertFalse(BiometricVectorIndex.objects.filter(template_id=retired.id).exists())

    def test_consolidate_move_keeps_the_side_row_linked_to_the_same_template(self):
        # No collision: fingerprint (template-kind, no side row) moves freely,
        # face's side row must still exist and now join to the kept subject.
        moved = enrol(SUBJECT_MODEL, "retired", "face", b"only-face-photo", actor="tester")

        consolidate(SUBJECT_MODEL, "kept", "retired", actor="tester")

        side = BiometricVectorIndex.objects.get(template_id=moved.id)
        side.template.refresh_from_db()
        self.assertEqual(side.template.subject_id, "kept")

    def test_reindex_backfills_from_a_row_created_without_the_orm(self):
        # Simulate data loaded before biometric_pgvector was installed: the
        # side row never gets created because BiometricTemplate.save() goes
        # through the ORM (so it WOULD sync) — bypass it the same way a raw
        # bulk load would, by deleting the side row the create() signal made.
        template = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")
        BiometricVectorIndex.objects.filter(template_id=template.id).delete()
        self.assertFalse(BiometricVectorIndex.objects.filter(template_id=template.id).exists())

        call_command("biometric_vector_reindex")

        self.assertTrue(BiometricVectorIndex.objects.filter(template_id=template.id).exists())

    def test_reindex_drops_orphan_side_rows(self):
        first = enrol(SUBJECT_MODEL, "s1", "face", b"photo-1", actor="tester")
        second = enrol(SUBJECT_MODEL, "s1", "face", b"photo-2", actor="tester")
        # first is now superseded but force a leftover side row onto it,
        # as if the supersede had raced ahead of the receiver.
        BiometricVectorIndex.objects.create(
            template_id=first.id, modality="face", provider="fake_embedding", model_name="", dim=1, embedding=[1.0],
        )

        call_command("biometric_vector_reindex")

        self.assertFalse(BiometricVectorIndex.objects.filter(template_id=first.id).exists())
        self.assertTrue(BiometricVectorIndex.objects.filter(template_id=second.id).exists())
