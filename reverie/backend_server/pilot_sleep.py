"""Recognize explicit bedtimes without overriding LLM-generated wake times."""
import re


def bedtime_hour(lifestyle):
    match = re.search(r"goes to bed (?:around |at )?(midnight|noon|\d{1,2}(?::\d{2})?\s*[ap]m)",
                      lifestyle, re.IGNORECASE)
    if not match:
        raise ValueError("Pilot requires an explicit, parseable bedtime in each lifestyle")
    value = match.group(1).lower().replace(" ", "")
    if value in ("midnight", "noon"):
        return 0 if value == "midnight" else 12
    hour = int(value[:-2].split(":")[0]) % 12
    return hour + (12 if value.endswith("pm") else 0)


def prewake_sleep(hour, wake_hour, lifestyle):
    """Preserve the old morning sleep block, except before a late bedtime.

    Evening bedtime remains a persona preference in the generated schedule;
    this introduces no global curfew or compulsory bedtime.
    """
    bedtime = bedtime_hour(lifestyle)
    start = bedtime if bedtime < wake_hour else 0
    return start <= hour < wake_hour
