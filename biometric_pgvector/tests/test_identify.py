"""
pgvector identify() branch (docs/wb-biometric-dedup-seam.md §6.2): must
return the same top-k (ids and order) as the NumPy path on one gallery, and
apply scope/self-exclusion/top_k identically.
"""

from biometric.apps import BiometricConfig
from biometric.models import BiometricTemplate
from biometric.services import identify
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase


class TestPgvectorIdentify(_MultimodalServiceTestCase):

    def _make_template(self, subject_id, vector, metadata=None):
        return BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="face",
            kind="embedding", vector=vector, provider="fake_embedding", model_name="",
            metadata=metadata or {},
        )

    def test_same_top_k_as_numpy_on_one_gallery(self):
        # Clear similarity margins so float4 rounding in pgvector cannot flip
        # the order against the NumPy (float64) path.
        self._make_template("closest", [1.0, 0.0, 0.0])
        self._make_template("middle", [0.7, 0.7, 0.0])
        self._make_template("farthest", [0.0, 1.0, 0.0])
        probe = [0.9, 0.1, 0.0]

        BiometricConfig.vector_index = "numpy"
        numpy_matches = identify("face", vector=probe, top_k=3)

        BiometricConfig.vector_index = "pgvector"
        pgvector_matches = identify("face", vector=probe, top_k=3)

        self.assertEqual([m.subject_id for m in numpy_matches], [m.subject_id for m in pgvector_matches])
        self.assertEqual([m.template_id for m in numpy_matches], [m.template_id for m in pgvector_matches])

    def test_scope_filters_on_metadata(self):
        self._make_template("in-scope", [1.0, 0.0], metadata={"cuvee_id": "A"})
        self._make_template("out-of-scope", [1.0, 0.0], metadata={"cuvee_id": "B"})

        BiometricConfig.vector_index = "pgvector"
        matches = identify("face", vector=[1.0, 0.0], scope={"cuvee_id": "A"})

        subject_ids = {m.subject_id for m in matches}
        self.assertIn("in-scope", subject_ids)
        self.assertNotIn("out-of-scope", subject_ids)

    def test_self_exclusion(self):
        self._make_template("self", [1.0, 0.0])
        self._make_template("other", [1.0, 0.0])

        BiometricConfig.vector_index = "pgvector"
        matches = identify("face", vector=[1.0, 0.0], exclude_subject="self")

        subject_ids = {m.subject_id for m in matches}
        self.assertNotIn("self", subject_ids)
        self.assertIn("other", subject_ids)

    def test_top_k_limits_results(self):
        for i in range(5):
            self._make_template(f"s{i}", [1.0, float(i) * 0.01])

        BiometricConfig.vector_index = "pgvector"
        matches = identify("face", vector=[1.0, 0.0], top_k=2)

        self.assertEqual(len(matches), 2)

    def test_superseded_template_excluded_from_gallery(self):
        active = self._make_template("s1", [1.0, 0.0])
        stale = self._make_template("s2", [1.0, 0.0])
        stale.validity_to = stale.date_created
        stale.save(update_fields=["validity_to"])

        BiometricConfig.vector_index = "pgvector"
        matches = identify("face", vector=[1.0, 0.0], top_k=5)

        subject_ids = {m.subject_id for m in matches}
        self.assertIn("s1", subject_ids)
        self.assertNotIn("s2", subject_ids)
