#!/usr/bin/env python3
"""Test A4: PresetManager — capture, save, load roundtrip, morph interpolation."""

import sys
import os
import tempfile

# Insert src/ into path so we can import from main and presets
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from main import LorenzAttractor, LogisticMap, RoesslerAttractor
from main import ManifoldMapper, VoicePool, CouplingField, DelayNetwork
from presets import PresetManager


def test_a4():
    pm = PresetManager()

    # ── 1. Build test objects ────────────────────────────────────────
    chaos = LorenzAttractor(sigma=10.0, rho=28.0, beta=2.667, dt=0.01)
    manifold = ManifoldMapper(n_centroids=16)
    pool = VoicePool(capacity=128, max_active=16)
    coupling = CouplingField()
    delay_net = DelayNetwork()
    macros = {
        'material': 0.5,
        'density': 0.6,
        'mutation': 0.3,
        'coherence': 0.7,
        'feedback': 0.5,
        'energy': 0.4,
    }

    # ── 2. capture() ─────────────────────────────────────────────────
    preset = pm.capture(chaos, manifold, pool, coupling, delay_net, macros)
    assert isinstance(preset, dict), "capture() must return a dict"
    assert preset['chaos']['type'] == 'LorenzAttractor'
    assert preset['chaos']['sigma'] == 10.0
    assert preset['chaos']['rho'] == 28.0
    assert preset['chaos']['beta'] == 2.667
    assert preset['chaos']['dt'] == 0.01
    assert preset['manifold']['n_centroids'] == 16
    assert preset['pool']['capacity'] == 128
    assert preset['pool']['max_active'] == 16
    assert preset['coupling'] == {'enabled': True}
    assert preset['delay_net']['wet_mix'] == 0.3
    assert preset['macros'] == macros
    print("  capture() ✓ — all fields present and correct")

    # ── 3. save() + load() roundtrip ─────────────────────────────────
    pm.save(preset, 'test_roundtrip')
    loaded = pm.load('test_roundtrip')

    # Verify all top-level keys
    for key in ['name', 'chaos', 'manifold', 'pool', 'coupling', 'delay_net', 'macros']:
        assert key in loaded, f"Missing key '{key}' in loaded preset"

    # Verify nested values
    assert loaded['chaos']['sigma'] == 10.0
    assert loaded['chaos']['rho'] == 28.0
    assert loaded['manifold']['n_centroids'] == 16
    assert loaded['pool']['capacity'] == 128
    assert loaded['macros']['material'] == 0.5
    assert loaded['name'] == 'test_roundtrip'
    print("  save() + load() roundtrip ✓")

    # ── 4. list_presets() ────────────────────────────────────────────
    presets = pm.list_presets()
    assert 'test_roundtrip' in presets
    assert isinstance(presets, list)
    print(f"  list_presets() ✓ — found {len(presets)} preset(s): {presets}")

    # ── 5. capture() with LogisticMap ────────────────────────────────
    chaos_logistic = LogisticMap(r=3.7)
    preset_log = pm.capture(chaos_logistic, manifold, pool, coupling,
                            delay_net, macros)
    assert preset_log['chaos']['type'] == 'LogisticMap'
    assert preset_log['chaos']['r'] == 3.7
    print("  capture(LogisticMap) ✓")

    # ── 6. capture() with RoesslerAttractor ──────────────────────────
    chaos_roessler = RoesslerAttractor(a=0.2, b=0.2, c=5.7, dt=0.03)
    preset_roe = pm.capture(chaos_roessler, manifold, pool, coupling,
                            delay_net, macros)
    assert preset_roe['chaos']['type'] == 'RoesslerAttractor'
    assert preset_roe['chaos']['a'] == 0.2
    print("  capture(RoesslerAttractor) ✓")

    # ── 7. morph() at t=0.5 ──────────────────────────────────────────
    pm.save(preset_log, 'test_logistic')
    pm.save(preset_roe, 'test_roessler')

    preset_a = pm.load('test_logistic')
    preset_b = pm.load('test_roessler')

    morphed = pm.morph(preset_a, preset_b, t=0.5)

    # Name is a combined morph string
    assert morphed['name'] == 'morph_test_logistic_test_roessler'

    # Chaos: type is string → picks A (since t=0.5 picks A for t<0.5 is false, picks B)
    # Actually: t=0.5 → t < 0.5 is False → picks B for strings
    assert morphed['chaos']['type'] == 'RoesslerAttractor', \
        f"At t=0.5, string should pick B: got {morphed['chaos']['type']}"

    # Verify morph at t=0 (should be A)
    morph_0 = pm.morph(preset_a, preset_b, t=0.0)
    assert morph_0['chaos']['type'] == 'LogisticMap'
    assert morph_0['pool']['capacity'] == 128  # both are 128, no change

    # Verify morph at t=1.0 (should be B)
    morph_1 = pm.morph(preset_a, preset_b, t=1.0)
    assert morph_1['chaos']['type'] == 'RoesslerAttractor'
    print("  morph() at t=0.0, t=0.5, t=1.0 ✓")

    # ── 8. morph numeric interpolation ───────────────────────────────
    # Create two presets with different numeric values
    chaos_a = LorenzAttractor(sigma=6.0, rho=20.0, beta=2.0, dt=0.005)
    preset_a2 = pm.capture(chaos_a, ManifoldMapper(n_centroids=8),
                           VoicePool(capacity=64, max_active=4),
                           coupling, delay_net, {'density': 0.2})

    chaos_b = LorenzAttractor(sigma=14.0, rho=36.0, beta=3.0, dt=0.015)
    preset_b2 = pm.capture(chaos_b, ManifoldMapper(n_centroids=32),
                           VoicePool(capacity=256, max_active=24),
                           coupling, delay_net, {'density': 0.8})

    morph_mid = pm.morph(preset_a2, preset_b2, t=0.5)

    # At t=0.5, numeric values should be midpoint
    assert abs(morph_mid['chaos']['sigma'] - 10.0) < 0.01, \
        f"sigma midpoint: expected 10.0, got {morph_mid['chaos']['sigma']}"
    assert abs(morph_mid['chaos']['rho'] - 28.0) < 0.01
    assert abs(morph_mid['chaos']['beta'] - 2.5) < 0.01
    assert abs(morph_mid['chaos']['dt'] - 0.01) < 0.001
    assert morph_mid['manifold']['n_centroids'] == 20  # (8+32)/2
    assert morph_mid['pool']['capacity'] == 160  # (64+256)/2
    assert morph_mid['pool']['max_active'] == 14  # (4+24)/2
    assert abs(morph_mid['macros']['density'] - 0.5) < 0.01
    print("  morph() numeric interpolation at t=0.5 ✓")

    print("\nPASS")


if __name__ == "__main__":
    try:
        test_a4()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nFAIL: {e}")
        sys.exit(1)
