import django.db.models.deletion
import pgvector.django
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("biometric", "0001_initial"),
    ]

    operations = [
        # Other apps may use the extension, so rolling this app back leaves it installed.
        migrations.RunSQL("CREATE EXTENSION IF NOT EXISTS vector", migrations.RunSQL.noop),
        migrations.CreateModel(
            name="BiometricVectorIndex",
            fields=[
                (
                    "template",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        primary_key=True,
                        related_name="vector_index",
                        serialize=False,
                        to="biometric.biometrictemplate",
                    ),
                ),
                ("modality", models.CharField(max_length=16)),
                ("provider", models.CharField(max_length=64)),
                ("model_name", models.CharField(max_length=64)),
                ("dim", models.IntegerField()),
                ("embedding", pgvector.django.VectorField()),
            ],
            options={
                "db_table": "biometric_vector_index",
            },
        ),
        migrations.AddIndex(
            model_name="biometricvectorindex",
            index=models.Index(
                fields=["modality", "provider", "model_name"],
                name="bvi_gallery_idx",
            ),
        ),
    ]
