"""Two simulated reference agents. No LLM, no network, fully deterministic."""

import random

from .data import label

GUESS_POOL = ["the usual arrangement", "the same as last time", "whatever the default is", "the typical option"]


def build_memory(scenario):
    """slot -> {"history": [values in write order], "restricted_to": list or None}."""
    mem = {}
    for msg in scenario["transcript"]:
        for fact in msg.get("facts", []):
            entry = mem.setdefault(fact["slot"], {"history": [], "restricted_to": None})
            entry["history"].append(fact["value"])
            if fact.get("restricted_to"):
                entry["restricted_to"] = fact["restricted_to"]
    return mem


class CarefulAgent:
    """Latest-write-wins; abstains when never stated; never leaks restricted facts."""
    name = "careful"

    def answer(self, scenario, probe):
        mem = build_memory(scenario)
        entry = mem.get(probe["slot"])
        if entry is None:
            return "I don't have any record of that."
        if entry["restricted_to"] and probe["asker_group"] not in entry["restricted_to"]:
            return "I can't share that detail with you."
        return "The " + label(probe["slot"]) + " is " + entry["history"][-1] + "."


class SloppyAgent:
    """Fixed degradation schedule keyed on scenario index (documented in README).

    current probes:    correct only when index % 3 == 0, otherwise recalls the FIRST (stale) value
    restricted probes: withheld only when index % 5 == 0, otherwise leaks the value
    unstated probes:   abstains only when index % 6 == 0, otherwise guesses (seeded by index)
    """
    name = "sloppy"

    def answer(self, scenario, probe):
        idx = scenario["index"]
        mem = build_memory(scenario)
        entry = mem.get(probe["slot"])
        kind = probe["kind"]
        if kind == "current" and entry is not None:
            if idx % 3 == 0:
                return "The " + label(probe["slot"]) + " is " + entry["history"][-1] + "."
            return "I have the " + label(probe["slot"]) + " as " + entry["history"][0] + "."
        if kind == "restricted" and entry is not None:
            if idx % 5 == 0 and probe["asker_group"] not in (entry["restricted_to"] or []):
                return "I can't share that detail with you."
            return "It's " + entry["history"][-1] + "."
        if idx % 6 == 0 and entry is None:
            return "I don't have any record of that."
        guess = GUESS_POOL[random.Random(idx * 1000 + len(probe["slot"])).randrange(len(GUESS_POOL))]
        return "I believe it's " + guess + "."
