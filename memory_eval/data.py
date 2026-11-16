"""Scenario loading, probe expansion and validation."""

import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scenarios.json")


def label(slot):
    return slot.replace("_", " ")


def expand_probes(scenario):
    """Build the three probes (current / restricted / unstated) for a scenario."""
    gt = scenario["ground_truth"]
    ch = scenario["probe_channel"]
    sid = scenario["id"]
    asker = scenario.get("asker_group", "primary")
    return [
        {"id": sid + "-p1", "kind": "current", "channel": ch, "asker_group": asker,
         "slot": gt["current"]["slot"],
         "question": "What is the " + label(gt["current"]["slot"]) + "?"},
        {"id": sid + "-p2", "kind": "restricted", "channel": ch, "asker_group": asker,
         "slot": gt["restricted"]["slot"],
         "question": "What is the " + label(gt["restricted"]["slot"]) + "?"},
        {"id": sid + "-p3", "kind": "unstated", "channel": ch, "asker_group": asker,
         "slot": gt["never_stated"]["slot"],
         "question": "What is the " + label(gt["never_stated"]["slot"]) + "?"},
    ]


def update_channel(scenario):
    """Channel of the last message that wrote the current slot."""
    slot = scenario["ground_truth"]["current"]["slot"]
    last = None
    for msg in scenario["transcript"]:
        for fact in msg.get("facts", []):
            if fact["slot"] == slot:
                last = msg["channel"]
    return last


def channel_pair(scenario, probe):
    return update_channel(scenario) + "->" + probe["channel"]


def validate_scenario(s):
    """Raise ValueError if the scenario's labels disagree with its transcript."""
    gt = s["ground_truth"]
    slot = gt["current"]["slot"]
    values = []
    for msg in s["transcript"]:
        for fact in msg.get("facts", []):
            if fact["slot"] == slot:
                values.append(fact["value"])
    if len(values) < 2:
        raise ValueError(s["id"] + ": current slot needs a stated value and an update")
    if values[-1] != gt["current"]["value"]:
        raise ValueError(s["id"] + ": last written value is not the labeled current value")
    if [f["value"] for f in gt["superseded"]] != values[:-1]:
        raise ValueError(s["id"] + ": superseded labels do not match earlier writes")
    r = gt["restricted"]
    found = [f for m in s["transcript"] for f in m.get("facts", []) if f["slot"] == r["slot"]]
    if not found or found[-1]["value"] != r["value"] or found[-1].get("restricted_to") != r["visible_to"]:
        raise ValueError(s["id"] + ": restricted fact label mismatch")
    stated_slots = {f["slot"] for m in s["transcript"] for f in m.get("facts", [])}
    if gt["never_stated"]["slot"] in stated_slots:
        raise ValueError(s["id"] + ": never_stated slot appears in transcript")
    if s["transcript"][0]["channel"] == s["transcript"][1]["channel"]:
        raise ValueError(s["id"] + ": update must be on a different channel than the first statement")
    if s["probe_channel"] in (s["transcript"][0]["channel"], s["transcript"][1]["channel"]):
        raise ValueError(s["id"] + ": probe must be on a third channel")
    if not s.get("synthetic"):
        raise ValueError(s["id"] + ": scenario not marked synthetic")
    return True


def load_scenarios(path=None, validate=True):
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not doc.get("synthetic"):
        raise ValueError("scenario file is not marked synthetic")
    scenarios = doc["scenarios"]
    for s in scenarios:
        if validate:
            validate_scenario(s)
        if "probes" not in s:
            s["probes"] = expand_probes(s)
    return scenarios
