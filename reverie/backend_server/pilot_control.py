"""Operational wall-clock limits; never part of agent state or simulation time."""
from datetime import datetime, timezone
from pathlib import Path


def aware_time(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Wall-clock deadlines require an explicit UTC offset")
    return parsed


class SessionPaused(Exception):
    pass


class SessionControl:
    def __init__(self, stop_at=None, survey_cutoff=None, stop_file=None, now=None):
        self.stop_at = aware_time(stop_at) if stop_at else None
        self.survey_cutoff = aware_time(survey_cutoff) if survey_cutoff else None
        self.stop_file = Path(stop_file) if stop_file else None
        self.now = now or (lambda: datetime.now(timezone.utc))
        if self.stop_at and self.survey_cutoff and self.survey_cutoff > self.stop_at:
            raise ValueError("Survey cutoff must not follow the simulation stop time")

    def check(self, phase):
        if self.stop_file and self.stop_file.exists():
            raise SessionPaused("stop_file")
        current = self.now()
        if self.stop_at and current >= self.stop_at:
            raise SessionPaused("wall_clock_stop")
        if phase == "survey" and self.survey_cutoff and current >= self.survey_cutoff:
            raise SessionPaused("survey_cutoff")
