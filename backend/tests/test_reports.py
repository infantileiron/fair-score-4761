import csv
import io
import unittest

from src.services import reports


class FakeRun:
    def __init__(self, result):
        self.result = result


class FakeReport:
    run_id = "run-1"


class FakeDb:
    def __init__(self, report, run=None):
        self._report = report
    def get(self, model, id):
        return self._report


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.report = FakeReport()
        self.result = {
            "groups": [{
                "group": "A", "n": 2, "selected": 1, "qualified": 1,
                "tp": 1, "fp": 0, "fn": 0, "tn": 1,
                "selection_rate": 0.5, "tpr": 1.0, "fpr": 0.0,
            }],
            "frontier": [{
                "accuracy": 0.75, "selection_gap": 0.125,
                "correct": 6, "n": 8, "pareto": True,
                "intervals": [
                    {"lower": 0, "upper": 12.5,
                     "lower_inclusive": True, "upper_inclusive": True},
                    {"lower": 40, "upper": 100,
                     "lower_inclusive": False, "upper_inclusive": True},
                ],
            }],
        }
        self._original = reports.require_completed
        reports.require_completed = lambda db, id: FakeRun(self.result)
    def tearDown(self):
        reports.require_completed = self._original

    def test_export_contains_group_counts_and_frontier_intervals(self):
        text = reports.export(FakeDb(self.report, None), "report-1")
        self.assertTrue(text.startswith("﻿"))
        section = text.lstrip("﻿").split("\n\n")
        groups = list(csv.DictReader(io.StringIO(section[0])))
        self.assertEqual(groups[0]["n"], "2")
        self.assertEqual(groups[0]["tp"], "1")
        frontier = list(csv.DictReader(io.StringIO(section[1])))
        self.assertEqual(frontier[0]["frontier_raw_correct"], "6")
        self.assertEqual(frontier[0]["frontier_raw_n"], "8")
        self.assertEqual(frontier[0]["frontier_threshold_intervals"],
                         "[0, 12.5] | (40, 100]")

    def test_export_legacy_result_without_intervals(self):
        self.result["frontier"] = [{
            "threshold": 65, "accuracy": 0.7, "selection_gap": 0.1,
            "pareto": True,
        }]
        text = reports.export(FakeDb(self.report, None), "report-1")
        frontier = list(csv.DictReader(io.StringIO(
            text.lstrip("﻿").split("\n\n")[1])))
        self.assertEqual(frontier[0]["frontier_threshold_intervals"], "阈值 65")
