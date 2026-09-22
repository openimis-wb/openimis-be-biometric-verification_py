import re

from django.core.management.base import BaseCommand, CommandError
from django.db import connection


def index_name(model_name, dim):
    """Deterministic, valid identifier for the per-model/dimension partial index."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", model_name).lower().strip("_") or "model"
    return f"biometric_vector_index_{safe}_{dim}_hnsw"


class Command(BaseCommand):
    help = (
        "Create (or drop) a partial HNSW index on biometric_vector_index for "
        "one model_name at a fixed dimension: "
        "CREATE INDEX ... USING hnsw ((embedding::vector(N)) vector_cosine_ops) "
        "WHERE model_name = 'NAME' (docs/wb-biometric-dedup-seam.md §6.2)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--model", dest="model_name", required=True, help="model_name to index.")
        parser.add_argument("--dim", type=int, required=True, help="Vector dimension to cast to for this model.")
        parser.add_argument("--drop", action="store_true", help="Drop the index instead of creating it.")

    def handle(self, *args, **options):
        model_name = options["model_name"]
        dim = options["dim"]
        if dim <= 0:
            raise CommandError("--dim must be a positive integer.")

        name = index_name(model_name, dim)
        with connection.cursor() as cursor:
            if options["drop"]:
                cursor.execute(f'DROP INDEX IF EXISTS "{name}"')
                self.stdout.write(self.style.SUCCESS(f"Dropped index {name}."))
                return

            cursor.execute(
                f'CREATE INDEX IF NOT EXISTS "{name}" ON biometric_vector_index '
                f"USING hnsw ((embedding::vector({dim})) vector_cosine_ops) "
                f"WHERE model_name = %s",
                [model_name],
            )
            self.stdout.write(self.style.SUCCESS(f"Index {name} ready."))
