from django.core.management.base import BaseCommand

from biometric.services import purge


class Command(BaseCommand):
    help = (
        "Erase biometric templates past BiometricRetentionPolicy's retention "
        "windows. No-op unless the policy has purge_enabled (superseded "
        "templates) or purge_active_enabled (active templates) set."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--actor", default="retention",
            help="Recorded as erased_by on the BiometricErasure tombstone(s).",
        )

    def handle(self, *args, **options):
        tombstone = purge(actor=options["actor"])
        if tombstone is None:
            self.stdout.write(self.style.NOTICE("Nothing purged (policy disabled, unset, or no stale templates)."))
            return
        self.stdout.write(self.style.SUCCESS(f"Purge complete. Last tombstone: {tombstone.id}."))
