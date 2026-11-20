"""Shared metadata preference rules; no suitability/constraint decisions here."""

from qdrant_client import models

from app.schemas import JobRecord, Preferences


def preference_warnings(job: JobRecord, preferences: Preferences) -> list[str]:
    warnings = []
    if preferences.work_mode and job.work_mode == "unknown":
        warnings.append("Work mode is unknown; verify the arrangement.")
    if preferences.location_city and job.location_city is None:
        if job.work_mode == "remote":
            warnings.append("India-wide remote role; no city restriction is listed.")
        else:
            warnings.append("City is unknown; verify the location.")
    return warnings


def job_passes_preferences(job: JobRecord, preferences: Preferences) -> bool:
    include_unknown = preferences.unknown_metadata_policy == "include_with_warning"
    if preferences.work_mode and job.work_mode != preferences.work_mode:
        if job.work_mode != "unknown" or not include_unknown:
            return False
    if preferences.location_city:
        if job.location_city is None:
            # A null-city remote job is explicitly India-wide, not missing location.
            if job.work_mode != "remote" and not include_unknown:
                return False
        elif job.location_city.strip().casefold() != preferences.location_city.strip().casefold():
            return False
    return True


def eligible_ids(jobs: list[JobRecord], preferences: Preferences) -> list[str]:
    return [job.job_id for job in jobs if job_passes_preferences(job, preferences)]


def qdrant_filter(ids: list[str]) -> models.Filter:
    # Use the same eligible ID set for every retrieval method in later modules.
    return models.Filter(must=[models.FieldCondition(
        key="metadata.job_id", match=models.MatchAny(any=ids),
    )])
