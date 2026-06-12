"""Tests for arena.engine.calibration (U3). Fully offline."""

from __future__ import annotations

import math
import random

import pytest

from arena.engine.calibration import (
    CalibrationLog,
    Calibrator,
    calibrate_signal,
    calibration_table,
    drift_flag,
    early_damping,
    log_score,
    overconfidence_flag,
)
from arena.models import Direction, Signal


def make_signal(
    direction: Direction = Direction.LONG,
    confidence: float = 0.9,
    p_long: float | None = None,
    p_short: float | None = None,
    p_flat: float | None = None,
) -> Signal:
    return Signal(
        agent="a",
        symbol="BTC",
        direction=direction,
        confidence=confidence,
        price_at_signal=100.0,
        p_long=p_long,
        p_short=p_short,
        p_flat=p_flat,
    )


def overconfident_history(n: int = 100, seed: int = 7) -> list[tuple[float, bool]]:
    """Agent always states 0.9 but only hits 60% of the time."""
    rng = random.Random(seed)
    hits = [True] * 60 + [False] * 40
    rng.shuffle(hits)
    return [(0.9, h) for h in hits[:n]]


# ---------------------------------------------------------------------------
# Calibrator — platt
# ---------------------------------------------------------------------------


class TestPlatt:
    def test_fixes_systematic_overconfidence(self) -> None:
        cal = Calibrator("platt")
        cal.fit(overconfident_history())
        calibrated = cal.apply(0.9)
        assert calibrated < 0.8
        assert abs(calibrated - 0.6) < 0.1  # pulled toward the empirical rate

    def test_output_in_unit_interval_and_monotone(self) -> None:
        rng = random.Random(0)
        history = [(p, rng.random() < p * 0.7) for p in
                   [rng.random() for _ in range(200)]]
        cal = Calibrator("platt")
        cal.fit(history)
        grid = [i / 20 for i in range(21)]
        out = [cal.apply(p) for p in grid]
        assert all(0.0 <= o <= 1.0 for o in out)
        # sigmoid(a*p+b) with a>0 is increasing; fitted on data where hit
        # probability rises with p, the map must be monotone non-decreasing.
        assert all(b >= a - 1e-9 for a, b in zip(out, out[1:]))

    def test_identity_under_min_samples(self) -> None:
        cal = Calibrator("platt")
        cal.fit([(0.9, False)] * 9)  # 9 < 10 samples
        assert cal.apply(0.9) == pytest.approx(0.9)
        assert cal.apply(0.3) == pytest.approx(0.3)

    def test_exactly_min_samples_fits(self) -> None:
        cal = Calibrator("platt")
        cal.fit([(0.9, i < 3) for i in range(10)])  # 30% hits at stated 0.9
        assert cal.apply(0.9) < 0.9


# ---------------------------------------------------------------------------
# Calibrator — isotonic
# ---------------------------------------------------------------------------


class TestIsotonic:
    def test_monotone(self) -> None:
        rng = random.Random(1)
        history = [(rng.random(), rng.random() < 0.5) for _ in range(300)]
        cal = Calibrator("isotonic")
        cal.fit(history)
        grid = [i / 50 for i in range(51)]
        out = [cal.apply(p) for p in grid]
        assert all(0.0 <= o <= 1.0 for o in out)
        assert all(b >= a - 1e-9 for a, b in zip(out, out[1:]))

    def test_recovers_step_pattern(self) -> None:
        # Below stated 0.5 the agent never hits; above it always hits.
        history = [(i / 100, i / 100 > 0.5) for i in range(100)]
        cal = Calibrator("isotonic")
        cal.fit(history)
        assert cal.apply(0.2) < 0.1
        assert cal.apply(0.9) > 0.9

    def test_identity_under_min_samples(self) -> None:
        cal = Calibrator("isotonic")
        cal.fit([(0.8, True)] * 5)
        assert cal.apply(0.8) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Calibrator — none / construction
# ---------------------------------------------------------------------------


class TestNoneMethod:
    def test_identity_even_with_data(self) -> None:
        cal = Calibrator("none")
        cal.fit(overconfident_history())
        for p in (0.1, 0.5, 0.9):
            assert cal.apply(p) == pytest.approx(p)

    def test_unknown_method_rejected(self) -> None:
        with pytest.raises(ValueError):
            Calibrator("magic")

    def test_apply_clamps_input(self) -> None:
        cal = Calibrator("none")
        assert cal.apply(1.5) == 1.0
        assert cal.apply(-0.2) == 0.0


# ---------------------------------------------------------------------------
# calibrate_signal
# ---------------------------------------------------------------------------


class TestCalibrateSignal:
    def fitted(self) -> Calibrator:
        cal = Calibrator("platt")
        cal.fit(overconfident_history())
        return cal

    def test_renormalized_and_direction_confidence_consistent(self) -> None:
        sig = make_signal(Direction.LONG, 0.8, p_long=0.8, p_short=0.15, p_flat=0.05)
        out = self.fitted().calibrate_signal(sig)
        assert out.has_probs
        assert out.p_long + out.p_short + out.p_flat == pytest.approx(1.0)
        probs = {
            Direction.LONG: out.p_long,
            Direction.SHORT: out.p_short,
            Direction.FLAT: out.p_flat,
        }
        assert out.direction == max(probs, key=probs.get)
        assert out.confidence == pytest.approx(max(probs.values()))

    def test_overconfident_vector_flattened(self) -> None:
        sig = make_signal(Direction.LONG, 0.9, p_long=0.9, p_short=0.05, p_flat=0.05)
        out = self.fitted().calibrate_signal(sig)
        assert out.p_long < sig.p_long  # confidence haircut
        assert out.direction == Direction.LONG  # ranking preserved

    def test_signal_without_probs_unchanged(self) -> None:
        sig = make_signal(Direction.SHORT, 0.7)
        out = self.fitted().calibrate_signal(sig)
        assert out == sig

    def test_unfitted_calibrator_is_identity(self) -> None:
        sig = make_signal(Direction.LONG, 0.9, p_long=0.9, p_short=0.05, p_flat=0.05)
        assert Calibrator("platt").calibrate_signal(sig) == sig

    def test_module_level_wrapper(self) -> None:
        sig = make_signal(Direction.LONG, 0.9, p_long=0.9, p_short=0.05, p_flat=0.05)
        cal = self.fitted()
        assert calibrate_signal(sig, cal) == cal.calibrate_signal(sig)


# ---------------------------------------------------------------------------
# early_damping
# ---------------------------------------------------------------------------


class TestEarlyDamping:
    def test_damping_math(self) -> None:
        sig = make_signal(Direction.LONG, 0.9, p_long=0.9, p_short=0.06, p_flat=0.04)
        out = early_damping(sig, rounds_seen=3, threshold=20)
        assert out.p_long == pytest.approx(0.7 * 0.9 + 0.3 / 3)
        assert out.p_short == pytest.approx(0.7 * 0.06 + 0.3 / 3)
        assert out.p_flat == pytest.approx(0.7 * 0.04 + 0.3 / 3)
        assert out.p_long + out.p_short + out.p_flat == pytest.approx(1.0)
        assert out.direction == Direction.LONG
        assert out.confidence == pytest.approx(out.p_long)

    def test_no_damping_at_threshold(self) -> None:
        sig = make_signal(Direction.LONG, 0.9, p_long=0.9, p_short=0.06, p_flat=0.04)
        assert early_damping(sig, rounds_seen=20, threshold=20) == sig
        assert early_damping(sig, rounds_seen=50, threshold=20) == sig

    def test_legacy_signal_confidence_damped(self) -> None:
        sig = make_signal(Direction.SHORT, 0.9)
        out = early_damping(sig, rounds_seen=0, threshold=20)
        assert out.confidence == pytest.approx(0.7 * 0.9 + 0.3 / 3)
        assert out.direction == Direction.SHORT
        assert not out.has_probs


# ---------------------------------------------------------------------------
# calibration_table / drift_flag / overconfidence_flag
# ---------------------------------------------------------------------------


class TestTableAndFlags:
    def test_table_bins(self) -> None:
        history = [(0.95, True)] * 4 + [(0.92, False)] * 1 + [(0.15, False)] * 5
        table = calibration_table(history, bins=10)
        assert len(table) == 2  # only populated bins reported
        low, high = table[0], table[1]
        assert low["n"] == 5 and low["p_mean"] == pytest.approx(0.15)
        assert low["hit_rate"] == 0.0
        assert high["n"] == 5
        assert high["p_mean"] == pytest.approx((0.95 * 4 + 0.92) / 5)
        assert high["hit_rate"] == pytest.approx(0.8)

    def test_drift_flag_on_miscalibrated_bin(self) -> None:
        # Stated ~0.9, hits only 50% -> gap 0.4 > 0.15.
        bad = [(0.9, i % 2 == 0) for i in range(40)]
        assert drift_flag(calibration_table(bad)) is True

    def test_no_drift_when_calibrated(self) -> None:
        good = [(0.9, i < 9) for i in range(10)] + [(0.5, i < 5) for i in range(10)]
        assert drift_flag(calibration_table(good)) is False

    def test_drift_gap_parameter(self) -> None:
        history = [(0.7, i < 5) for i in range(10)]  # gap = 0.2
        table = calibration_table(history)
        assert drift_flag(table, gap=0.15) is True
        assert drift_flag(table, gap=0.25) is False

    def test_overconfidence_flag(self) -> None:
        assert overconfidence_flag([(0.9, i % 3 == 0) for i in range(30)]) is True
        # High p but also high hit rate -> not overconfident.
        assert overconfidence_flag([(0.9, i % 3 != 0) for i in range(30)]) is False
        # Low stated p -> not overconfident even when missing.
        assert overconfidence_flag([(0.4, False) for _ in range(30)]) is False
        assert overconfidence_flag([]) is False


# ---------------------------------------------------------------------------
# log_score
# ---------------------------------------------------------------------------


class TestLogScore:
    def test_basic(self) -> None:
        sig = make_signal(Direction.LONG, 0.5, p_long=0.5, p_short=0.3, p_flat=0.2)
        assert log_score(sig, Direction.LONG) == pytest.approx(math.log(0.5))
        assert log_score(sig, Direction.FLAT) == pytest.approx(math.log(0.2))

    def test_clamped_at_minus_six(self) -> None:
        sig = make_signal(Direction.LONG, 1.0, p_long=1.0, p_short=0.0, p_flat=0.0)
        assert log_score(sig, Direction.SHORT) == -6.0
        tiny = make_signal(
            Direction.LONG, 0.999, p_long=0.9999, p_short=0.0001, p_flat=0.0
        )
        assert log_score(tiny, Direction.SHORT) == -6.0

    def test_perfect_call_scores_zero(self) -> None:
        sig = make_signal(Direction.LONG, 1.0, p_long=1.0, p_short=0.0, p_flat=0.0)
        assert log_score(sig, Direction.LONG) == pytest.approx(0.0)

    def test_legacy_signal_uses_confidence(self) -> None:
        sig = make_signal(Direction.SHORT, 0.8)
        assert log_score(sig, Direction.SHORT) == pytest.approx(math.log(0.8))
        assert log_score(sig, Direction.LONG) == pytest.approx(math.log(0.1))


# ---------------------------------------------------------------------------
# CalibrationLog
# ---------------------------------------------------------------------------


class TestCalibrationLog:
    def test_round_trip(self, tmp_path) -> None:
        db = tmp_path / "arena.db"
        log = CalibrationLog(str(db))
        log.add("claude", 0.9, True, 1)
        log.add("claude", 0.7, False, 2)
        log.add("gpt", 0.5, True, 1)
        assert log.history("claude", window=10) == [(0.9, True), (0.7, False)]
        assert log.history("gpt", window=10) == [(0.5, True)]
        assert log.history("missing", window=10) == []
        log.close()
        # Persisted across reopen.
        log2 = CalibrationLog(str(db))
        assert log2.history("claude", window=10) == [(0.9, True), (0.7, False)]
        log2.close()

    def test_window_keeps_most_recent(self, tmp_path) -> None:
        log = CalibrationLog(str(tmp_path / "a.db"))
        for rid in range(20):
            log.add("a", rid / 20, rid % 2 == 0, rid)
        hist = log.history("a", window=5)
        assert len(hist) == 5
        assert hist == [(rid / 20, rid % 2 == 0) for rid in range(15, 20)]
        assert log.history("a", window=0) == []
        log.close()

    def test_feeds_calibrator(self, tmp_path) -> None:
        log = CalibrationLog(str(tmp_path / "b.db"))
        for rid, (p, hit) in enumerate(overconfident_history()):
            log.add("a", p, hit, rid)
        cal = Calibrator("platt")
        cal.fit(log.history("a", window=100))
        assert cal.apply(0.9) < 0.8
        log.close()
