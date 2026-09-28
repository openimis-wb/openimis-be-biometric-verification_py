from django.core.management.base import BaseCommand, CommandError

from biometric.audit_chain import check_anchor, store_chain_check, verify_chain

# checked_by of the BiometricAuditChainCheck rows this command stores.
COMMAND_ACTOR = "biometric_audit_verify"


class Command(BaseCommand):
    help = (
        "Walk the biometric audit chain from its first event and report the first "
        "divergence. Stores the walk as the audit chain status (biometricAuditChainStatus). "
        "Prints the head sequence and hash on every run: store them "
        "outside this database and pass them back as --expected-sequence / "
        "--expected-head, the only check that reveals a truncated tail. The chain "
        "may have grown since; the event at that sequence must still hold that hash. "
        "A recorded head that no longer holds fails the command; the stored status "
        "covers the walk only."
    )

    def add_arguments(self, parser):
        parser.add_argument("--batch-size", type=int, default=1000, help="Rows read per database round trip.")
        parser.add_argument("--expected-head", default=None, help="Head hash recorded out of band by an earlier run.")
        parser.add_argument("--expected-sequence", type=int, default=None, help="Head sequence recorded out of band by an earlier run.")

    def handle(self, *args, **options):
        report = verify_chain(batch_size=options["batch_size"])
        store_chain_check(report, actor=COMMAND_ACTOR)
        if report.divergence is not None:
            divergence = report.divergence
            raise CommandError(
                f"biometric audit chain diverges after {report.checked} event(s): "
                f"{divergence.kind} at sequence {divergence.sequence}: {divergence.detail}"
            )

        problem = check_anchor(
            report, sequence=options["expected_sequence"], head_hash=options["expected_head"],
        )
        if problem is not None:
            raise CommandError(f"biometric audit chain does not match the recorded head: {problem}")

        self.stdout.write(
            f"biometric audit chain intact over {report.checked} event(s); "
            f"head sequence {report.head_sequence} hash {report.head_hash}"
        )
