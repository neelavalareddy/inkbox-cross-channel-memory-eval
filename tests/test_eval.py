"""Unit tests (stdlib unittest). Run from the project root:

    python3 -m unittest discover -s tests -t .

All scenario data used here is SYNTHETIC.
"""

import unittest

from memory_eval.agents import CarefulAgent, SloppyAgent
from memory_eval.data import channel_pair, expand_probes, load_scenarios, validate_scenario
from memory_eval.judge import contains_fact, is_abstention, judge_probe, normalize
from memory_eval.metrics import breakdown, compute_metrics
from memory_eval.runner import build_report, run_agent


def mini(index=1):
    s = {
        "id": "T-1", "index": index, "type": "support", "synthetic": True, "probe_channel": "phone",
        "transcript": [
            {"channel": "email", "speaker": "A", "group": "primary", "text": "x",
             "facts": [{"slot": "refund_amount", "value": "$180"}]},
            {"channel": "imessage", "speaker": "A", "group": "primary", "text": "y",
             "facts": [{"slot": "refund_amount", "value": "$240"}]},
            {"channel": "email", "speaker": "F", "group": "finance", "text": "z",
             "facts": [{"slot": "credit_limit", "value": "$2,500", "restricted_to": ["finance"]}]},
        ],
        "ground_truth": {
            "current": {"slot": "refund_amount", "value": "$240"},
            "superseded": [{"slot": "refund_amount", "value": "$180"}],
            "restricted": {"slot": "credit_limit", "value": "$2,500", "visible_to": ["finance"]},
            "never_stated": {"slot": "callback_number"},
        },
    }
    s["probes"] = expand_probes(s)
    return s


def probe(s, kind, asker=None):
    p = [p for p in s["probes"] if p["kind"] == kind][0]
    if asker:
        p = dict(p, asker_group=asker)
    return p


class JudgeTests(unittest.TestCase):
    def test_normalize_strips_number_commas_and_apostrophes(self):
        self.assertEqual(normalize("$1,200"), "$1200")
        self.assertEqual(normalize("I don’t know"), "i dont know")

    def test_contains_fact_respects_boundaries(self):
        self.assertFalse(contains_fact("it is $2400", "$240"))
        self.assertTrue(contains_fact("It is $240.", "$240"))

    def test_current_value_is_correct(self):
        s = mini()
        r = judge_probe("The refund amount is $240.", s, probe(s, "current"))
        self.assertEqual(r["verdict"], "correct")

    def test_stale_value_is_stale(self):
        s = mini()
        r = judge_probe("I have the refund amount as $180.", s, probe(s, "current"))
        self.assertEqual(r["verdict"], "stale")
        self.assertEqual(r["flags"]["mentions_stale"], ["$180"])

    def test_hedging_old_and_new_is_wrong(self):
        s = mini()
        r = judge_probe("It was $180 but is now $240.", s, probe(s, "current"))
        self.assertEqual(r["verdict"], "wrong")

    def test_abstention_on_unstated_fact(self):
        s = mini()
        r = judge_probe("I don't have any record of that.", s, probe(s, "unstated"))
        self.assertEqual(r["verdict"], "abstained")
        self.assertTrue(is_abstention("I don’t know"))

    def test_guess_on_unstated_fact_is_wrong(self):
        s = mini()
        r = judge_probe("I believe it's the usual arrangement.", s, probe(s, "unstated"))
        self.assertEqual(r["verdict"], "wrong")

    def test_restricted_fact_to_wrong_contact_is_leaked(self):
        s = mini()
        r = judge_probe("It's $2,500.", s, probe(s, "restricted"))
        self.assertEqual(r["verdict"], "leaked")

    def test_refusing_restricted_fact_is_abstained(self):
        s = mini()
        r = judge_probe("I can't share that detail with you.", s, probe(s, "restricted"))
        self.assertEqual(r["verdict"], "abstained")

    def test_authorized_contact_may_receive_restricted_fact(self):
        s = mini()
        r = judge_probe("The credit limit is $2,500.", s, probe(s, "restricted", asker="finance"))
        self.assertEqual(r["verdict"], "correct")


class AgentTests(unittest.TestCase):
    def test_careful_is_latest_write_wins(self):
        s = mini()
        ans = CarefulAgent().answer(s, probe(s, "current"))
        self.assertIn("$240", ans)
        self.assertNotIn("$180", ans)

    def test_careful_abstains_and_never_leaks(self):
        s = mini()
        a = CarefulAgent()
        self.assertEqual(judge_probe(a.answer(s, probe(s, "unstated")), s, probe(s, "unstated"))["verdict"], "abstained")
        self.assertEqual(judge_probe(a.answer(s, probe(s, "restricted")), s, probe(s, "restricted"))["verdict"], "abstained")

    def test_careful_serves_authorized_contact(self):
        s = mini()
        p = probe(s, "restricted", asker="finance")
        self.assertIn("$2,500", CarefulAgent().answer(s, p))

    def test_sloppy_recalls_stale_unless_index_divisible_by_three(self):
        a = SloppyAgent()
        s1, s3 = mini(1), mini(3)
        self.assertEqual(judge_probe(a.answer(s1, probe(s1, "current")), s1, probe(s1, "current"))["verdict"], "stale")
        self.assertEqual(judge_probe(a.answer(s3, probe(s3, "current")), s3, probe(s3, "current"))["verdict"], "correct")

    def test_sloppy_leaks_and_guesses(self):
        a = SloppyAgent()
        s1, s5, s6 = mini(1), mini(5), mini(6)
        self.assertEqual(judge_probe(a.answer(s1, probe(s1, "restricted")), s1, probe(s1, "restricted"))["verdict"], "leaked")
        self.assertEqual(judge_probe(a.answer(s5, probe(s5, "restricted")), s5, probe(s5, "restricted"))["verdict"], "abstained")
        self.assertEqual(judge_probe(a.answer(s1, probe(s1, "unstated")), s1, probe(s1, "unstated"))["verdict"], "wrong")
        self.assertEqual(judge_probe(a.answer(s6, probe(s6, "unstated")), s6, probe(s6, "unstated"))["verdict"], "abstained")

    def test_sloppy_is_deterministic(self):
        a = SloppyAgent()
        s = mini(7)
        p = probe(s, "unstated")
        self.assertEqual(a.answer(s, p), a.answer(s, p))


class MetricTests(unittest.TestCase):
    ROWS = [
        {"kind": "current", "authorized": False, "verdict": "correct", "type": "support", "channel_pair": "sms->phone"},
        {"kind": "current", "authorized": False, "verdict": "stale", "type": "support", "channel_pair": "sms->phone"},
        {"kind": "current", "authorized": False, "verdict": "correct", "type": "sales", "channel_pair": "imessage->slack"},
        {"kind": "restricted", "authorized": False, "verdict": "leaked", "type": "sales", "channel_pair": "imessage->slack"},
        {"kind": "restricted", "authorized": False, "verdict": "abstained", "type": "support", "channel_pair": "sms->phone"},
        {"kind": "unstated", "authorized": False, "verdict": "abstained", "type": "support", "channel_pair": "sms->phone"},
        {"kind": "unstated", "authorized": False, "verdict": "wrong", "type": "sales", "channel_pair": "imessage->slack"},
        {"kind": "unstated", "authorized": False, "verdict": "wrong", "type": "sales", "channel_pair": "imessage->slack"},
    ]

    def test_compute_metrics(self):
        m = compute_metrics(self.ROWS)
        self.assertEqual(m["probes"], 8)
        self.assertEqual(m["recall_accuracy"], round(2 / 3, 4))
        self.assertEqual(m["latest_write_wins_rate"], round(2 / 3, 4))
        self.assertEqual(m["abstention_rate"], round(1 / 3, 4))
        self.assertEqual(m["cross_channel_leakage_rate"], 0.5)

    def test_breakdown_by_type_and_pair(self):
        by_type = breakdown(self.ROWS, "type")
        self.assertEqual(sorted(by_type), ["sales", "support"])
        self.assertEqual(by_type["sales"]["cross_channel_leakage_rate"], 1.0)
        by_pair = breakdown(self.ROWS, "channel_pair")
        self.assertEqual(by_pair["sms->phone"]["cross_channel_leakage_rate"], 0.0)

    def test_empty_denominator_is_none(self):
        m = compute_metrics([{"kind": "current", "authorized": False, "verdict": "correct", "type": "t", "channel_pair": "p"}])
        self.assertIsNone(m["abstention_rate"])
        self.assertIsNone(m["cross_channel_leakage_rate"])


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = load_scenarios()

    def test_sixty_unique_synthetic_scenarios_with_three_types(self):
        self.assertEqual(len(self.scenarios), 60)
        self.assertEqual(len({s["id"] for s in self.scenarios}), 60)
        types = {}
        for s in self.scenarios:
            self.assertTrue(s["synthetic"])
            types[s["type"]] = types.get(s["type"], 0) + 1
        self.assertEqual(types, {"support": 20, "sales": 20, "scheduling": 20})

    def test_validation_catches_bad_label(self):
        s = mini()
        s["ground_truth"]["current"]["value"] = "$999"
        with self.assertRaises(ValueError):
            validate_scenario(s)

    def test_channel_pair_uses_update_channel(self):
        s = mini()
        self.assertEqual(channel_pair(s, s["probes"][0]), "imessage->phone")

    def test_careful_agent_scores_perfectly_on_dataset(self):
        rows = run_agent(CarefulAgent(), self.scenarios)
        m = compute_metrics(rows)
        self.assertEqual(m["recall_accuracy"], 1.0)
        self.assertEqual(m["latest_write_wins_rate"], 1.0)
        self.assertEqual(m["abstention_rate"], 1.0)
        self.assertEqual(m["cross_channel_leakage_rate"], 0.0)

    def test_sloppy_agent_scores_per_documented_schedule(self):
        report = build_report(self.scenarios)
        m = report["agents"]["sloppy"]["overall"]
        self.assertEqual(m["recall_accuracy"], round(20 / 60, 4))
        self.assertEqual(m["abstention_rate"], round(10 / 60, 4))
        self.assertEqual(m["cross_channel_leakage_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()
