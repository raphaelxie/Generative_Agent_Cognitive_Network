"""Single-hour output contract for the pilot; never accept another hour's row."""
import json


class PilotScheduleError(BaseException):
    """Stop instead of turning malformed schedule outputs into sleeping agents."""


def parse_hourly_activity(raw, requested_hour):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hour", "activity"}:
        raise ValueError("Expected exactly hour and activity")
    if value["hour"] != requested_hour:
        raise ValueError("Response belongs to a different hour")
    activity = value["activity"]
    if isinstance(activity, str) and ("\n" in activity or "\r" in activity):
        raise ValueError("Expected a single-line activity")
    if isinstance(activity, str):
        activity = activity.strip().rstrip(".").strip()
    if (not isinstance(activity, str) or not 3 <= len(activity.strip()) <= 150
            or "\n" in activity or "\r" in activity or not any(c.isalnum() for c in activity)):
        raise ValueError("Expected one short activity phrase")
    return activity


def build_hourly_prompt(persona, requested_hour, previous, hour_strings, intermission=None):
    context = {"identity": persona.scratch.get_str_iss(),
               "date": persona.scratch.get_str_curr_date_str(),
               "requested_hour": requested_hour,
               "daily_plan": persona.scratch.daily_req,
               "previous_hours": [{"hour": hour, "activity": activity}
                                  for hour, activity in zip(hour_strings, previous)],
               "sleep_instruction": intermission}
    return ("Plan ONE hour of this person's day, consistent with their identity, intended daily "
            "plan, prior activities, and sleep/wake routine. The requested_hour is the hour to "
            "answer, even when earlier hours appear in context. Return ONLY one JSON object "
            "with exactly two keys: hour (copy requested_hour exactly) and activity (one short "
            "verb phrase, 3 to 150 characters). Do not return the entire day, a timetable, "
            "markdown, or an explanation. A midnight activity is not an answer for an afternoon hour.\n"
            + json.dumps(context, ensure_ascii=False))


def generate_hourly_activity(persona, requested_hour, previous, hour_strings, intermission=None):
    from pilot_tasks import request_contract
    prompt = build_hourly_prompt(persona, requested_hour, previous, hour_strings, intermission)
    return request_contract(prompt, lambda raw: parse_hourly_activity(raw, requested_hour),
                            "Hourly schedule", max_tokens=512), prompt


def parse_daily_plan(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"activities"}:
        raise ValueError("Expected exactly activities")
    activities = value.get("activities") if isinstance(value, dict) else None
    if not isinstance(activities, list) or not 3 <= len(activities) <= 24:
        raise ValueError("Daily plan must contain 3 to 24 activities")
    if any(isinstance(a, str) and ("\n" in a or "\r" in a) for a in activities):
        raise ValueError("Daily activities must be single-line strings")
    activities = [a.strip().rstrip(".").strip() if isinstance(a, str) else a for a in activities]
    if any(not isinstance(a, str) or not 3 <= len(a) <= 250
           or "\n" in a or "\r" in a or not any(c.isalnum() for c in a) for a in activities):
        raise ValueError("Daily activities must be short nonempty strings")
    return activities


def generate_daily_plan(persona, wake_up_hour):
    from pilot_tasks import request_contract
    context = {"identity": persona.scratch.get_str_iss(),
               "lifestyle": persona.scratch.get_str_lifestyle(),
               "date": persona.scratch.get_str_curr_date_str(),
               "wake_up_hour": wake_up_hour}
    prompt = ("Generate this person's complete daily plan in broad strokes, consistent with "
              "their identity and lifestyle. Include times for activities and respect the "
              "generated wake_up_hour and any bedtime after midnight. Return ONLY JSON "
              "with one key, activities: a list of 3 to 24 short activity strings, each "
              "including a time or time range. Cover the day's main activities and bedtime; "
              "a wake-up item alone is not a complete plan. Do not return a numbered list "
              "or prose outside JSON.\n" + json.dumps(context, ensure_ascii=False))
    return request_contract(prompt, parse_daily_plan, "Daily plan"), prompt


def parse_wake_hour(raw):
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {"wake_hour"}
            or type(value["wake_hour"]) is not int or not 0 <= value["wake_hour"] <= 23):
        raise ValueError("Expected wake_hour as an integer from 0 to 23")
    return value["wake_hour"]


def generate_wake_hour(persona):
    from pilot_tasks import request_contract
    context = {"identity": persona.scratch.get_str_iss(),
               "lifestyle": persona.scratch.get_str_lifestyle(),
               "date": persona.scratch.get_str_curr_date_str()}
    prompt = ('Choose today\'s wake-up hour from this person\'s lifestyle and plans. '
              'Return ONLY JSON {"wake_hour": integer from 0 to 23}. Use a 24-hour clock.\n'
              + json.dumps(context, ensure_ascii=False))
    return request_contract(prompt, parse_wake_hour, "Wake hour", max_tokens=128), prompt
