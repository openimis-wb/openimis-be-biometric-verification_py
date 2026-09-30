"""
The subject a GraphQL caller acts on or reads, checked against the user's
location scope (docs/wb-biometric-dedup-seam.md §6.15).

A subject model with a relation to location.Location and a
get_queryset(queryset, user) classmethod (individual.Individual) is scoped
by that classmethod, the one its own list applies. Any other subject model is
only checked for existence. Superusers and IMIS admins bypass the scope, as
core does.
"""

from django.apps import apps
from django.core.exceptions import ValidationError
from django.db.models import CharField, Q
from django.db.models.functions import Cast

SUBJECT_MODEL_UNKNOWN = "BIOMETRIC_SUBJECT_MODEL_UNKNOWN"
SUBJECT_NOT_FOUND = "BIOMETRIC_SUBJECT_NOT_FOUND"


class SubjectRefusedError(ValueError):
    """
    The subject model is not an installed model, or the subject does not exist
    or lies outside the caller's scope. Both of the latter carry
    SUBJECT_NOT_FOUND, as core's lists show neither.
    """

    def __init__(self, code, message):
        # graphql-core copies .extensions onto the GraphQL error it reports.
        self.extensions = {"code": code}
        super().__init__(message)


def resolve_subject_model(label):
    """The model class a subject_model label names; SubjectRefusedError when none is installed."""
    try:
        return apps.get_model(label)
    except (LookupError, ValueError, TypeError):
        raise SubjectRefusedError(SUBJECT_MODEL_UNKNOWN, f"Unknown subject model '{label}'.") from None


def is_location_scoped(model):
    """True when the model relates to location.Location and scopes its own queryset by user."""
    if not apps.is_installed("location") or not callable(getattr(model, "get_queryset", None)):
        return False
    location_model = apps.get_model("location", "Location")
    return any(
        field.is_relation and field.related_model is location_model
        for field in model._meta.get_fields()
    )


def bypasses_scope(user):
    """Superusers and IMIS admins see every subject, as in core's scoped querysets."""
    return bool(getattr(user, "is_superuser", False) or getattr(user, "is_imis_admin", False))


def check_subject(user, subject_model, subject_id):
    """
    SubjectRefusedError unless subject_model is an installed model, subject_id
    one of its rows, and that row within the user's scope.
    """
    model = resolve_subject_model(subject_model)
    not_found = SubjectRefusedError(SUBJECT_NOT_FOUND, f"No {subject_model} '{subject_id}' in scope.")
    try:
        queryset = model.objects.filter(pk=subject_id)
        if is_location_scoped(model) and not bypasses_scope(user):
            queryset = model.get_queryset(queryset, user)
        exists = queryset.exists()
    except (ValidationError, ValueError, TypeError):
        raise not_found from None
    if not exists:
        raise not_found


def scope_rows(queryset, user, *, keep_unattributed=False, live_subjects=False):
    """
    Rows of a biometric table (subject_model, subject_id) whose subject the
    user may see. Rows of a location-scoped subject model are kept when their
    subject is in the model's scoped queryset; rows of an unscoped model are
    kept; rows naming no installed model are dropped. With keep_unattributed,
    rows whose subject_id is empty name no subject and are kept. With
    live_subjects, a row is kept only when its subject row exists and, for a
    model with is_deleted, is not soft-deleted, whether the model is scoped or not.
    """
    if bypasses_scope(user):
        return queryset
    visible = Q(subject_id="") if keep_unattributed else Q(pk__in=[])
    for label in queryset.order_by().values_list("subject_model", flat=True).distinct():
        try:
            model = resolve_subject_model(label)
        except SubjectRefusedError:
            continue
        if is_location_scoped(model):
            subjects = model.get_queryset(None, user)
        elif live_subjects:
            subjects = model._default_manager.all()
        else:
            visible |= Q(subject_model=label)
            continue
        if live_subjects and _is_soft_deletable(model):
            subjects = subjects.filter(is_deleted=False)
        scoped_ids = subjects.annotate(_subject_id=Cast("pk", CharField())).values("_subject_id")
        visible |= Q(subject_model=label, subject_id__in=scoped_ids)
    return queryset.filter(visible)


def _is_soft_deletable(model):
    return any(field.name == "is_deleted" for field in model._meta.concrete_fields)


def require_rows_in_scope(queryset, user, *, keep_unattributed=False):
    """
    SubjectRefusedError(SUBJECT_NOT_FOUND) when the queryset holds rows and none
    of them is in the user's scope (scope_rows). An empty queryset passes: the
    caller reports the missing row itself.
    """
    if queryset.exists() and not scope_rows(queryset, user, keep_unattributed=keep_unattributed).exists():
        raise SubjectRefusedError(SUBJECT_NOT_FOUND, "No such row in scope.")
