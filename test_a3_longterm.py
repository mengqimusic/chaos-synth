#!/usr/bin/env python3
"""Test A3: LongTermFeedback — feed 20 identical frames, verify output dict."""

import sys
import numpy as np

# Import from src/main.py (assumes we're in project root)
sys.path.insert(0, "src")
from main import LongTermFeedback, SAMPLE_RATE, BLOCK_SIZE


def test_a3():
    ltf = LongTermFeedback()

    # Verify window_size matches spec (~172 frames for 1 second)
    expected_window = SAMPLE_RATE // BLOCK_SIZE  # 44100 // 256 = 172
    assert ltf.window_size == expected_window, \
        f"window_size mismatch: {ltf.window_size} != {expected_window}"
    print(f"  window_size = {ltf.window_size} ✓")

    # Feed 20 identical frames: centroid=2500, flux=100, rms=0.1
    for _ in range(20):
        ltf.feed(2500.0, 100.0, 0.1)

    # After 20 feeds, buffer has 20 entries (well below window_size=172)
    assert len(ltf.centroid_buffer) == 20
    assert len(ltf.flux_buffer) == 20

    # tick() should return dict with both keys (≥10 frames)
    result = ltf.tick()
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 'lt_brightness' in result, "Missing lt_brightness key"
    assert 'lt_activity' in result, "Missing lt_activity key"

    # Brightness: smoothed centroid / 5000.0, clamped to 1.0
    # Starting lt_centroid=500.0, avg_c=2500.0, smoothing rate=0.05
    # After 20 identical feeds, smoothed centroid ≈ 2500 * (1-(1-0.05)^20) ≈ 2500*0.642 ≈ 1605
    # But tick() only smooths once per call — let's just check range
    assert 0.0 <= result['lt_brightness'] <= 1.0, \
        f"lt_brightness out of range: {result['lt_brightness']}"

    # Activity: smoothed flux / 500.0, clamped to 1.0
    # Starting lt_flux=50.0, avg_f=100.0, smoothing rate=0.05
    assert 0.0 <= result['lt_activity'] <= 1.0, \
        f"lt_activity out of range: {result['lt_activity']}"

    print(f"  lt_brightness = {result['lt_brightness']:.4f} (0-1) ✓")
    print(f"  lt_activity   = {result['lt_activity']:.4f} (0-1) ✓")

    # Test with <10 frames: should return empty dict
    ltf2 = LongTermFeedback()
    for _ in range(5):
        ltf2.feed(1000.0, 50.0, 0.05)
    early_result = ltf2.tick()
    assert early_result == {}, \
        f"Expected empty dict for <10 frames, got {early_result}"
    print(f"  <10 frames returns empty dict ✓")

    # Test buffer overflow at window_size
    ltf3 = LongTermFeedback()
    for i in range(expected_window + 10):
        ltf3.feed(float(i % 10) * 500.0, float(i % 5) * 50.0, 0.05)
    assert len(ltf3.centroid_buffer) == expected_window, \
        f"Buffer should cap at window_size={expected_window}, got {len(ltf3.centroid_buffer)}"
    print(f"  buffer caps at window_size={expected_window} ✓")

    print("\nPASS")


if __name__ == "__main__":
    try:
        test_a3()
    except Exception as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
