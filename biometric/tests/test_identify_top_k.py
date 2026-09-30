"""identifyBiometric clamps topK to 1..BIOMETRIC["MAX_TOP_K"] (default 50)."""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import graphene

from biometric.apps import DEFAULT_CFG, BiometricConfig
from biometric.schema import Mutation, Query
from biometric.services import enrol
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase


class TestTopKClamp(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        self._max_top_k = BiometricConfig.max_top_k
        self.addCleanup(setattr, BiometricConfig, "max_top_k", self._max_top_k)
        for i in range(6):
            enrol(SUBJECT_MODEL, f"s{i}", "face", b"photo-%d" % i, actor="agent")

    def _count(self, top_k):
        user = MagicMock(is_anonymous=False, username="agent")
        user.has_perms.return_value = True
        sample = base64.b64encode(b"photo-0").decode()
        argument = "" if top_k is None else ", topK: %d" % top_k
        result = graphene.Schema(query=Query, mutation=Mutation).execute(
            'query { identifyBiometric(modality: "face", sample: "%s"%s) { subjectId } }' % (sample, argument),
            context_value=SimpleNamespace(user=user, headers={}),
        )
        self.assertIsNone(result.errors, result.errors)
        return len(result.data["identifyBiometric"])

    def test_default_is_fifty(self):
        self.assertEqual(DEFAULT_CFG["max_top_k"], 50)
        self.assertEqual(BiometricConfig.max_top_k, 50)

    def test_negative_and_zero_rank_one(self):
        self.assertEqual(self._count(-3), 1)
        self.assertEqual(self._count(0), 1)

    def test_above_the_cap_ranks_the_cap(self):
        BiometricConfig.max_top_k = 4
        self.assertEqual(self._count(1000), 4)

    def test_omitted_ranks_five(self):
        self.assertEqual(self._count(None), 5)
