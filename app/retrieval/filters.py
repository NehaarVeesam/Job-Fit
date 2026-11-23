"""Apply candidate preferences to job document metadata."""

from langchain_core.documents import Document

from app.schemas import Preferences


def matches_preferences(
    document: Document,
    preferences: Preferences | None,
) -> bool:
    """Return whether a job satisfies every supplied search filter."""
    if preferences is None:
        return True

    desired_city = (preferences.location_city or "").strip().casefold()
    job_city = (document.metadata.get("location_city") or "").strip().casefold()

    if desired_city and job_city != desired_city:
        return False

    return (
        preferences.work_mode is None
        or document.metadata.get("work_mode") == preferences.work_mode
    )
