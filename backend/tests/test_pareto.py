import math
import time
import unittest
from fractions import Fraction

from src.services.analysis import analyze
from src.services.pareto import (
    attainable_decisions,
    frontier_result,
    format_intervals,
    mark_pareto,
    merge_decisions,
    threshold_frontier,
)


def make_rows(groups):
    """groups: {name: [(score, qualified), ...]} -> row dicts."""
    rows = []
    for name, observations in groups.items():
        for index, (score, qualified) in enumerate(observations):
            rows.append({
                "candidate_id": f"{name}-{index}",
                "group": name,
                "score": score,
                "qualified": qualified,
            })
    return rows


def merged_points(rows):
    return frontier_result(rows)["points"]


class AttainableThresholdTests(unittest.TestCase):
    def test_covers_zero_to_hundred_without_grid_gaps(self):
        rows = make_rows({
            "A": [(0, 0), (12.25, 1), (47.5, 1), (99.99, 0)],
            "B": [(0, 1), (12.25, 0), (47.5, 0), (99.99, 1)],
        })
        decisions = attainable_decisions(rows)
        self.assertEqual(len(decisions), 5)
        self.assertEqual(decisions[0]["interval"], {
            "lower": 99.99, "upper": 100,
            "lower_inclusive": False, "upper_inclusive": True,
        })
        self.assertEqual(decisions[-1]["interval"], {
            "lower": 0, "upper": 0,
            "lower_inclusive": True, "upper_inclusive": True,
        })

    def test_decimal_scores_are_exact_threshold_boundaries(self):
        rows = make_rows({
            "A": [(64.99, 1), (65.01, 0)],
            "B": [(64.99, 0), (65.01, 0)],
        })
        decisions = attainable_decisions(rows)
        bounds = [d["interval"]["lower"] for d in decisions]
        self.assertIn(64.99, bounds)
        self.assertIn(65.01, bounds)
        self.assertEqual(len(decisions), 3)

    def test_zero_and_hundred_scores(self):
        rows = make_rows({
            "A": [(0, 0), (100, 1)],
            "B": [(0, 0), (100, 0)],
        })
        decisions = attainable_decisions(rows)
        # No selection set exists for thresholds above 100.
        self.assertEqual(len(decisions), 2)
        # Thresholds in (0, 100] still select the perfect score, so 100 is
        # included rather than producing an empty decision above the maximum.
        top = decisions[0]
        self.assertEqual(top["interval"]["upper"], 100)
        self.assertTrue(top["interval"]["upper_inclusive"])
        self.assertEqual(top["interval"]["lower"], 0)
        self.assertFalse(top["interval"]["lower_inclusive"])
        self.assertEqual(top["correct"], 3)

    def test_ties_cross_together(self):
        rows = make_rows({
            "A": [(80, 1), (80, 0), (40, 1)],
            "B": [(80, 0), (40, 0), (40, 1)],
        })
        decisions = attainable_decisions(rows)
        # Two distinct scores -> three selection sets, ties never split.
        self.assertEqual(len(decisions), 3)
        only_top = decisions[1]
        self.assertEqual(only_top["interval"], {
            "lower": 40, "upper": 80,
            "lower_inclusive": False, "upper_inclusive": True,
        })
        # Both 80s crossed together; raw counts show one correct, two wrong.
        self.assertEqual(only_top["correct"], 2)

    def test_unequal_group_sizes_raw_counts(self):
        rows = make_rows({
            "A": [(90, 1), (70, 0), (50, 0), (30, 0)],
            "B": [(90, 0), (70, 1)],
        })
        decisions = attainable_decisions(rows)
        self.assertEqual(decisions[0]["gap"], Fraction(0, 1))  # nobody selected
        top = decisions[1]  # only the 90-point scores cross
        self.assertEqual(top["correct"], 4)  # A:90 tp + 3 non-selected tn, B:90 fp
        self.assertEqual(top["n"], 6)
        # A selects 1/4, B selects 1/2; raw counts, not a shared denominator.
        self.assertEqual(top["gap"], Fraction(1, 4))


class DominanceTests(unittest.TestCase):
    def test_dominance_uses_exact_counts_not_rounded_rates(self):
        from src.services.pareto import mark_pareto
        # 1333/4000 and 1334/4000 both display as 33.3%; dominance must still
        # use the raw counts, so the higher-count point wins.
        interval = {"lower": 0, "upper": 100,
                    "lower_inclusive": True, "upper_inclusive": True}
        shared = dict(n=4000, intervals=[interval], pareto=True,
                      selection_gap=0.1, selection_gap_fraction=[1, 10])
        lower = dict(shared, correct=1332, accuracy=1332 / 4000)
        higher = dict(shared, correct=1333, accuracy=1333 / 4000)
        self.assertEqual(f"{lower['accuracy']*100:.1f}", f"{higher['accuracy']*100:.1f}")
        from src.services.pareto import _decision
        self.assertEqual(len(merge_decisions([
            dict(shared, correct=1332, accuracy=1332 / 4000,
                 gap=Fraction(1, 10), interval=interval),
            dict(shared, correct=1333, accuracy=1333 / 4000,
                 gap=Fraction(1, 10), interval=interval),
        ])), 2)
        points = mark_pareto([
            {"correct": 1332, "n": 4000, "accuracy": 1332 / 4000,
             "selection_gap": 0.1, "selection_gap_fraction": [1, 10],
             "intervals": [interval], "pareto": True},
            {"correct": 1333, "n": 4000, "accuracy": 1333 / 4000,
             "selection_gap": 0.1, "selection_gap_fraction": [1, 10],
             "intervals": [interval], "pareto": True},
        ])
        self.assertEqual([p["correct"] for p in points if p["pareto"]], [1333])

    def test_only_non_dominated_points_are_persisted(self):
        rows = make_rows({
            "A": [(10, 1), (20, 1), (30, 0)],
            "B": [(10, 0), (20, 0), (30, 0)],
        })
        result = frontier_result(rows)
        self.assertEqual(result["meta"]["frontier_points"],
                         len(result["points"]))
        self.assertTrue(all(p["pareto"] for p in result["points"]))
        self.assertGreaterEqual(result["meta"]["unique_metric_points"],
                                result["meta"]["frontier_points"])
        self.assertTrue(all(p["correct"] != 2 for p in result["points"]))

    def test_rounding_does_not_create_or_remove_dominance(self):
        rows = make_rows({
            "A": [(10, 1), (20, 0), (30, 0)],
            "B": [(10, 0), (20, 0), (30, 1)],
            "C": [(10, 0), (20, 1), (30, 0)],
        })
        from src.services.pareto import merge_decisions
        decisions = attainable_decisions(rows)
        merged = merge_decisions(decisions)
        keys = [(p["correct"], tuple(p["selection_gap_fraction"])) for p in merged]
        self.assertEqual(len(keys), len(set(keys)))
        # Nothing was dropped by rounding: merged intervals cover every
        # attainable decision exactly once.
        self.assertEqual(sum(len(p["intervals"]) for p in merged), len(decisions))
        for point in merged:
            self.assertAlmostEqual(point["accuracy"], point["correct"] / point["n"])

    def test_intervals_cover_entire_domain_without_gaps(self):
        rows = make_rows({
            "A": [(0, 0), (12.25, 1), (47.5, 1), (99.99, 0)],
            "B": [(0, 1), (12.25, 0), (47.5, 0), (99.99, 1)],
        })
        intervals = [d["interval"] for d in attainable_decisions(rows)]
        intervals.sort(key=lambda i: (i["lower"], not i["lower_inclusive"]))
        self.assertEqual(intervals[0]["lower"], 0)
        self.assertTrue(intervals[0]["lower_inclusive"])
        self.assertEqual(intervals[-1]["upper"], 100)
        self.assertTrue(intervals[-1]["upper_inclusive"])
        for lower, upper in zip(intervals, intervals[1:]):
            self.assertEqual(lower["upper"], upper["lower"])
            # One side must include the shared boundary; identical decisions
            # are merged, so both inclusive would be ambiguous.
            self.assertNotEqual(lower["upper_inclusive"], upper["lower_inclusive"])

    def test_equal_accuracy_equal_gap_are_merged(self):
        rows = make_rows({
            "A": [(3, 0), (5, 0)],
            "B": [(8, 1), (8, 1)],
        })
        merged = merged_points(rows)
        flat = [p for p in merged if p["correct"] == 2 and p["selection_gap"] == 0]
        self.assertEqual(len(flat), 1)
        self.assertEqual(len(flat[0]["intervals"]), 2)
        self.assertLess(flat[0]["intervals"][0]["upper"], 100)

    def test_disconnected_intervals_preserved(self):
        rows = make_rows({
            "A": [(3, 0), (5, 0)],
            "B": [(8, 1), (8, 1)],
        })
        merged = merged_points(rows)
        point = next(p for p in merged if p["correct"] == 2)
        intervals = point["intervals"]
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0]["lower_inclusive"], True)
        self.assertEqual(intervals[1]["lower_inclusive"], False)
        # The two intervals are separated; the gap must remain visible.
        self.assertLess(intervals[0]["upper"], intervals[1]["lower"])

    def test_contiguous_intervals_are_coalesced(self):
        rows = make_rows({
            "A": [(10, 1), (20, 0), (30, 0), (40, 0)],
            "B": [(10, 0), (20, 1), (30, 1), (40, 1)],
        })
        merged = merged_points(rows)
        for point in merged:
            for left, right in zip(point["intervals"], point["intervals"][1:]):
                self.assertGreater(right["lower"], left["upper"])

    def test_frontier_strong_dominance(self):
        rows = make_rows({
            "A": [(10, 1), (20, 1), (30, 0)],
            "B": [(10, 0), (20, 0), (30, 0)],
        })
        from src.services.pareto import _fraction
        decisions = attainable_decisions(rows)
        all_points = mark_pareto(merge_decisions(decisions))
        points = merged_points(rows)
        self.assertTrue(points)
        for p in points:
            for q in all_points:
                if q is p:
                    continue
                dominates = (
                    q["correct"] >= p["correct"]
                    and _fraction(q) <= _fraction(p)
                    and (q["correct"] > p["correct"] or _fraction(q) < _fraction(p))
                )
                self.assertFalse(dominates)
        self.assertTrue(points)


class ScaleTests(unittest.TestCase):
    def test_full_sample_is_not_quadratic(self):
        groups = {"A": [], "B": [], "C": []}
        for index in range(5000):
            name = ("A", "B", "C")[index % 3]
            groups[name].append((index / 50, index % 2))
        rows = make_rows(groups)
        started = time.perf_counter()
        result = frontier_result(rows)
        elapsed = time.perf_counter() - started
        self.assertGreater(len(rows), 0)
        self.assertLess(elapsed, 1.0)
        self.assertGreater(result["meta"]["attainable_decisions"], 90)
        # At most k+1 selection sets for k distinct scores.
        self.assertLessEqual(
            result["meta"]["attainable_decisions"],
            len({row["score"] for row in rows}) + 1,
        )


class ExportFormatTests(unittest.TestCase):
    def test_interval_text_keeps_every_range_and_endpoint(self):
        point = {
            "intervals": [
                {"lower": 0, "upper": 12.5,
                 "lower_inclusive": True, "upper_inclusive": True},
                {"lower": 40, "upper": 100,
                 "lower_inclusive": False, "upper_inclusive": True},
            ]
        }
        self.assertEqual(format_intervals(point), "[0, 12.5] | (40, 100]")

    def test_legacy_point_still_renders_threshold(self):
        self.assertEqual(format_intervals({"threshold": 65}), "阈值 65")


class UpgradeCompatibilityTests(unittest.TestCase):
    def test_legacy_fifth_step_result_is_untouched(self):
        # Simulates a run persisted before the upgrade: no coverage metadata,
        # fixed-step points with only threshold/accuracy/gap/pareto fields.
        legacy = {
            "threshold": 65,
            "frontier": [
                {"threshold": t, "accuracy": 0.7, "selection_gap": 0.1, "pareto": t == 65}
                for t in range(0, 101, 5)
            ],
        }
        self.assertEqual(len(legacy["frontier"]), 21)
        self.assertNotIn("frontier_coverage", legacy)
        for point in legacy["frontier"]:
            self.assertNotIn("intervals", point)
            self.assertNotIn("correct", point)
        self.assertEqual(legacy["frontier"][13]["threshold"], 65)

    def test_new_result_exposes_both_metrics_and_intervals(self):
        rows = make_rows({
            "A": [(0, 0)] * 5 + [(50.5, 1)] * 5 + [(80, 0)] * 5 + [(100, 1)] * 5,
            "B": [(0, 1)] * 5 + [(50.5, 0)] * 5 + [(80, 1)] * 5 + [(100, 0)] * 5,
        })
        # make_rows derives unique ids from position, so no duplicates arise.
        result = analyze(rows, 50.5, 0.1)
        self.assertEqual(result["frontier_coverage"]["mode"], "attainable")
        distinct = len({row["score"] for row in rows})
        # max score is 100, so no empty decision above it exists.
        self.assertEqual(
            result["frontier_coverage"]["attainable_decisions"], distinct
        )
        for point in result["frontier"]:
            self.assertIsInstance(point["accuracy"], float)
            self.assertIsInstance(point["selection_gap"], float)
            self.assertTrue(point["intervals"])
            for interval in point["intervals"]:
                self.assertIsInstance(interval["lower"], (int, float))
                self.assertIsInstance(interval["upper"], (int, float))
