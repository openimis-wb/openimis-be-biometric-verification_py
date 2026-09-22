"""
Django signals for biometric verification module.

Automatically updates Claim risk scores when facial audits are created/modified.
"""

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import ClaimFacialAudit

logger = logging.getLogger(__name__)


@receiver(post_save, sender=ClaimFacialAudit)
def update_claim_risk_score_on_audit(sender, instance, created, **kwargs):
    """
    Signal handler: Automatically recalculate claim fraud risk score
    whenever a ClaimFacialAudit is created or updated.

    Stores the risk assessment in claim.json_ext['biometric_risk_assessment'].

    Args:
        sender: The model class (ClaimFacialAudit)
        instance: The ClaimFacialAudit instance that was saved
        created: Boolean - True if this is a new record
        **kwargs: Additional signal arguments

    Note: ClaimFacialAudit is an immutable audit trail - no soft-delete.
    """
    from .services import BiometricService

    claim_id = instance.claim_id
    action = "created" if created else "updated"

    try:
        # Calculate the global risk score for this claim
        risk_data = BiometricService.calculate_global_risk_score(claim_id)

        # Store risk assessment in Claim.json_ext
        from claim.models import Claim

        claim = Claim.objects.get(id=claim_id)

        # Initialize json_ext if it doesn't exist
        if claim.json_ext is None:
            claim.json_ext = {}

        # Store the complete risk assessment
        claim.json_ext['biometric_risk_assessment'] = {
            'risk_score': risk_data['risk_score'],
            'risk_level': risk_data['risk_level'],
            'audit_count': risk_data['audit_count'],
            'failed_audits': risk_data['failed_audits'],
            'score_variance': risk_data['score_variance'],
            'avg_similarity': risk_data['avg_similarity'],
            'last_updated': instance.audit_date.isoformat(),
        }

        claim.save()

        logger.info(
            f"Audit {action} for Claim {claim_id} - "
            f"Risk score updated: {risk_data['risk_score']} "
            f"({risk_data['risk_level']}) - stored in json_ext"
        )

    except Exception as e:
        logger.error(
            f"Failed to update risk score for Claim {claim_id} "
            f"after audit {action}: {e}",
            exc_info=True
        )
