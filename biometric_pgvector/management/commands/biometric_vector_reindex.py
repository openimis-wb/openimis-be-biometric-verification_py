from django.core.management.base import BaseCommand

from biometric.models import BiometricTemplate
from biometric_pgvector.models import BiometricVectorIndex
from biometric_pgvector.receivers import sync_one


class Command(BaseCommand):
    help = (
        "Backfill biometric_vector_index from biometric_template: replays the "
        "sync used by post_save/post_delete over every template, so an active "
        "embedding row gets its side row upserted and every other row loses "
        "one it should not have (fresh install, or after bulk data loading "
        "that bypassed the ORM signals)."
    )

    def handle(self, *args, **options):
        synced = 0
        for template in BiometricTemplate.objects.all().iterator():
            sync_one(template)
            synced += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Reindexed {synced} template(s); "
                f"{BiometricVectorIndex.objects.count()} vector(s) now indexed."
            )
        )
