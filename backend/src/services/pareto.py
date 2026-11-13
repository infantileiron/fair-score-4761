"""Every attainable threshold decision and its Pareto frontier.

A candidate is selected when ``score >= threshold`` and the threshold domain
is [0, 100].  With ``k`` distinct observed scores there are only ``k + 1``
selection sets:

* no score selected:       ``(max_score, 100]``  (empty, and omitted, when max_score == 100)
* top ``j`` score levels:  ``(next_lower_score, current_score]``
* every score selected:    ``[0, min_score]``

Candidates with the same score therefore cross the threshold together, and a
decision is recorded once per distinct score, not once per candidate.

Dominance is decided on raw sample counts (``correct`` and the exact gap
fraction), never on rounded percentages.  Decisions whose accuracy and
selection gap are identical are merged into one point that keeps *all* of
their threshold intervals, including disjoint ones.
"""
from collections import Counter, defaultdict
from fractions import Fraction


def _decision(correct, total, selected, group_sizes, group_names, interval):
    rates = [Fraction(selected[name], group_sizes[name]) for name in group_names]
    gap = max(rates) - min(rates) if len(rates) > 1 else Fraction(0, 1)
    return {
        "correct": correct,
        "n": total,
        "accuracy": correct / total,
        "selection_gap": float(gap),
        "selection_gap_fraction": [gap.numerator, gap.denominator],
        "gap": gap,
        "interval": interval,
    }


def attainable_decisions(rows):
    """Return one decision per attainable selection set.

    The sweep is ``O(n log n + k * groups)``: scores are sorted once and each
    observation is folded into its score level once.  Equal scores are
    consumed as a single block before a decision is recorded, so tied
    candidates always cross the line together.
    """
    total = len(rows)
    group_names = sorted({row["group"] for row in rows})
    group_sizes = Counter(row["group"] for row in rows)
    positives = Counter(row["group"] for row in rows if row["qualified"] == 1)
    # With nothing selected every negative is already classified correctly.
    correct = sum(group_sizes[name] - positives[name] for name in group_names)
    selected = Counter()
    by_score = defaultdict(list)
    for row in rows:
        by_score[row["score"]].append(row)
    scores = sorted(by_score, reverse=True)

    decisions = []
    if scores[0] < 100:
        decisions.append(_decision(
            correct, total, selected, group_sizes, group_names,
            {
                "lower": scores[0],
                "upper": 100,
                "lower_inclusive": False,
                "upper_inclusive": True,
            },
        ))

    for index, score in enumerate(scores):
        for row in by_score[score]:
            selected[row["group"]] += 1
            correct += 1 if row["qualified"] == 1 else -1
        if index + 1 < len(scores):
            interval = {
                "lower": scores[index + 1],
                "upper": score,
                "lower_inclusive": False,
                "upper_inclusive": True,
            }
        else:
            interval = {
                "lower": 0,
                "upper": score,
                "lower_inclusive": True,
                "upper_inclusive": True,
            }
        decisions.append(_decision(
            correct, total, selected, group_sizes, group_names, interval,
        ))
    return decisions


def _sort_key(interval):
    return (interval["lower"], interval["upper"], not interval["lower_inclusive"])


def merge_decisions(decisions):
    """Merge identical (accuracy, selection gap) decisions.

    The key is the raw correct count plus the reduced gap fraction, so float
    rounding can neither split equal points nor hide a dominance relation.
    """
    merged = {}
    for decision in decisions:
        key = (decision["correct"], _fraction(decision))
        item = merged.get(key)
        if item is None:
            item = {
                "correct": decision["correct"],
                "n": decision["n"],
                "accuracy": decision["accuracy"],
                "selection_gap": decision["selection_gap"],
                "selection_gap_fraction": decision["selection_gap_fraction"],
                "intervals": [],
                "pareto": True,
            }
            merged[key] = item
        item["intervals"].append(decision["interval"])
    points = list(merged.values())
    for point in points:
        point["intervals"] = _coalesce(point["intervals"])
    points.sort(key=lambda point: (-point["correct"], _fraction(point)))
    return points


def _coalesce(intervals):
    """Sort intervals and join neighbours that share a threshold.

    Two ranges ``(a, b]`` and ``(b, c]`` describe the same continuous set as
    ``(a, c]`` because exactly one of them owns ``b``; only ranges with a real
    gap between them stay separate.
    """
    ordered = sorted(intervals, key=_sort_key)
    merged = []
    for interval in ordered:
        if merged:
            previous = merged[-1]
            touches = previous["upper"] == interval["lower"]
            no_hole = previous["upper_inclusive"] or interval["lower_inclusive"]
            if touches and no_hole:
                previous["upper"] = interval["upper"]
                previous["upper_inclusive"] = interval["upper_inclusive"]
                continue
        merged.append(dict(interval))
    return merged


def _fraction(point):
    numerator, denominator = point["selection_gap_fraction"]
    return Fraction(numerator, denominator)


def mark_pareto(points):
    """Mark non-dominated points in O(m log m), without pairwise checks.

    Sorted by accuracy descending and gap ascending, only the smallest-gap
    point at each accuracy level can be non-dominated, and it survives only
    when its gap is strictly below every point of higher accuracy.
    """
    ordered = sorted(points, key=lambda point: (-point["correct"], _fraction(point)))
    best_gap_at_higher_accuracy = None
    index = 0
    while index < len(ordered):
        correct = ordered[index]["correct"]
        level_gap = _fraction(ordered[index])
        if best_gap_at_higher_accuracy is None or level_gap < best_gap_at_higher_accuracy:
            ordered[index]["pareto"] = True
        else:
            ordered[index]["pareto"] = False
        best_gap_at_higher_accuracy = (
            level_gap
            if best_gap_at_higher_accuracy is None
            else min(best_gap_at_higher_accuracy, level_gap)
        )
        index += 1
        while index < len(ordered) and ordered[index]["correct"] == correct:
            ordered[index]["pareto"] = False
            index += 1
    return ordered


def frontier_result(rows):
    """Return the non-dominated points plus coverage metadata.

    Every non-dominated point keeps *all* threshold intervals that share its
    accuracy and selection gap, including disjoint ones.  Dominated points are
    omitted from the payload; the metadata still reports how many attainable
    decisions and merged metric points were evaluated.
    """
    decisions = attainable_decisions(rows)
    points = mark_pareto(merge_decisions(decisions))
    frontier = [point for point in points if point["pareto"]]
    return {
        "mode": "attainable",
        "points": frontier,
        "meta": {
            "attainable_decisions": len(decisions),
            "unique_metric_points": len(points),
            "frontier_points": len(frontier),
        },
    }


def threshold_frontier(rows):
    """Backward-compatible entry point returning the merged points."""
    return frontier_result(rows)["points"]


def format_intervals(point):
    """Render a point's threshold intervals with explicit open/closed ends."""
    intervals = point.get("intervals")
    if not intervals:
        return "阈值 " + str(point.get("threshold", ""))
    return " | ".join(
        "{}{}, {}{}".format(
            "[" if interval["lower_inclusive"] else "(",
            interval["lower"],
            interval["upper"],
            "]" if interval["upper_inclusive"] else ")",
        )
        for interval in intervals
    )
