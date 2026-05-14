#!/usr/bin/env python3
"""chaos-synth — Granular Synthesis Engine (Phase 4)

Classic granular synthesis: manual parameters + LFO modulation + Timbre Map.
2520 exciter×body×modulator combos selectable via 8 arrangement maps.
Direct control — no chaos engine, no Voronoi, no feedback analysis.

Run: python src/ui/app.py
"""

import sys, os, threading
import numpy as np

# ── Import engine ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from main import (
    SAMPLE_RATE, BLOCK_SIZE, TimbreMap, LFO,
    VoicePool, SelfSampleBuffer, DelayNetwork,
    EXCITERS, BODIES,
)

import dearpygui.dearpygui as dpg
import sounddevice as sd

# ═══════════════════════════════════════════════════════════════════════
# Shared State
# ═══════════════════════════════════════════════════════════════════════
_params_lock = threading.Lock()
_params = {
    # Particle
    'rate': 20.0,           # grains per second
    'pitch': 440.0,         # base frequency Hz
    'size': 0.05,           # grain length seconds
    'feedback': 0.3,        # self-sample mix + delay
    # Spread
    'pitch_spread': 0.1,    # ±octaves
    'pan_spread': 0.5,      # 0=center, 1=full width
    'position_spread': 0.1, # combo position jitter
    # Timbre Map
    'timbre_map': 'Gradual', # map name
    'position': 0.5,        # 0→1 scan through map
    # LFO targets
    'lfo0_target': 'Pitch', 'lfo0_wave': 'Sine', 'lfo0_rate': 0.5, 'lfo0_depth': 0.0,
    'lfo1_target': 'Size',  'lfo1_wave': 'Sine', 'lfo1_rate': 0.3, 'lfo1_depth': 0.0,
    'lfo2_target': 'Position','lfo2_wave': 'Sine','lfo2_rate': 0.2, 'lfo2_depth': 0.0,
}

# ── Engine instances ───────────────────────────────────────────────────
_timbre_map = TimbreMap()
_pool = VoicePool(capacity=128, max_active=32)
_ssb = SelfSampleBuffer(duration_s=2.0)
_delay_net = DelayNetwork()
_lfos = [LFO("LFO1"), LFO("LFO2"), LFO("LFO3")]

# ── Callback state (audio thread only) ─────────────────────────────────
_cb_state = {'accumulator': 0.0}

_audio_stream = None

# ═══════════════════════════════════════════════════════════════════════
# Audio Callback
# ═══════════════════════════════════════════════════════════════════════
def _audio_callback(outdata, frames, time_info, status):
    if status:
        print(f"audio status: {status}", flush=True)
    outdata.fill(0.0)

    with _params_lock:
        p = dict(_params)

    dt = frames / SAMPLE_RATE

    # ── Sync LFO params + tick ─────────────────────────────────────────
    for i, lfo in enumerate(_lfos):
        key = f'lfo{i}'
        # Only sync if depth > 0 (avoid sync overhead when LFO unused)
        lfo.target = p[f'{key}_target']
        lfo.waveform = p[f'{key}_wave']
        lfo.rate = p[f'{key}_rate']
        lfo.depth = p[f'{key}_depth']
        lfo.tick(dt)

    # ── Apply LFO to parameters ────────────────────────────────────────
    def lfo_val(target):
        for lfo in _lfos:
            if lfo.target == target and lfo.depth > 0.001:
                return lfo.value * lfo.depth
        return 0.0

    rate = max(1.0, p['rate'] * (1.0 + lfo_val('Rate')))
    pitch = max(20.0, p['pitch'] * (1.0 + lfo_val('Pitch')))
    size = max(0.003, min(0.5, p['size'] * (1.0 + lfo_val('Size'))))
    position = np.clip(p['position'] + lfo_val('Position') * 0.5, 0.0, 1.0)
    pan_spread_lfo = np.clip(p['pan_spread'] + lfo_val('Pan'), 0.0, 1.0)

    # ── How many grains this block? ────────────────────────────────────
    _cb_state['accumulator'] += rate * dt
    n_grains = int(_cb_state['accumulator'])
    _cb_state['accumulator'] -= n_grains

    # ── Trigger grains ─────────────────────────────────────────────────
    for _ in range(n_grains):
        # Position → TimbreMap combo
        ppos = position + np.random.uniform(-p['position_spread'], p['position_spread'])
        eid, bid, mid = _timbre_map.get_combo(np.clip(ppos, 0.0, 1.0))

        # Pitch with spread
        pfreq = pitch * (2.0 ** np.random.uniform(-p['pitch_spread'], p['pitch_spread']))
        pfreq = np.clip(pfreq, 20.0, 12000.0)

        # Grain size in samples
        psize_samples = max(4, int(size * SAMPLE_RATE))

        # Amplitude (dynamic range based on voice count)
        amp = 0.3 / max(1.0, n_grains ** 0.5)

        # Trigger voice (VoicePool handles exciter→body→modulator internally)
        _pool.trigger(eid, bid, mid, pfreq, amp, 0.5)

        # Override grain buffer: truncate to desired size + apply self-sample mix
        for v in _pool.voices:
            if v['active'] and v['age'] == 0:
                grain = v['buffer']
                # Truncate to size
                g_slice = grain[:psize_samples]
                # Self-sample feedback mix
                if p['feedback'] > 0.001:
                    ssb_grain = _ssb.snatch(length=len(g_slice))
                    mix = min(len(g_slice), len(ssb_grain))
                    g_slice = g_slice[:mix] * (1.0 - p['feedback'] * 0.6) + ssb_grain[:mix] * p['feedback'] * 0.6
                # Write back (pad if needed)
                v['buffer'][:len(g_slice)] = g_slice
                if len(g_slice) < len(v['buffer']):
                    v['buffer'][len(g_slice):] = 0.0
                v['duration'] = len(g_slice)
                # Store pan for per-voice stereo spread
                v['pan'] = 0.5 + np.random.uniform(-1.0, 1.0) * pan_spread_lfo * 0.5
                break

    # ── Render pool with per-voice pan ─────────────────────────────────
    _pool.render(outdata.T)

    # ── Self-sample write + delay ──────────────────────────────────────
    _ssb.write(outdata.T.copy())
    _delay_net.set_feedback(p['feedback'])
    _delay_net.wet_mix = 0.05 + p['feedback'] * 0.7
    outdata[:] = _delay_net.process(outdata.T).T

# ═══════════════════════════════════════════════════════════════════════
# Audio Stream Control
# ═══════════════════════════════════════════════════════════════════════
def _start_audio():
    global _audio_stream
    _audio_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=2,
        dtype='float32',
        callback=_audio_callback,
    )
    _audio_stream.start()
    print("Audio stream started.")

def _stop_audio():
    global _audio_stream
    if _audio_stream is not None:
        _audio_stream.stop()
        _audio_stream.close()
        _audio_stream = None
        print("Audio stream stopped.")

# ═══════════════════════════════════════════════════════════════════════
# UI Callbacks
# ═══════════════════════════════════════════════════════════════════════
def _on_param_change(sender, app_data, user_data):
    with _params_lock:
        _params[user_data] = app_data

def _on_map_change(sender, app_data):
    _timbre_map.set_map(app_data)
    with _params_lock:
        _params['timbre_map'] = app_data

# ═══════════════════════════════════════════════════════════════════════
# Parameter definitions
# ═══════════════════════════════════════════════════════════════════════
LFO_TARGETS = ['Pitch', 'Size', 'Rate', 'Position', 'Pan']
LFO_WAVES = LFO.WAVEFORMS  # ['Sine', 'Triangle', 'Square', 'Saw', 'Random']

# ═══════════════════════════════════════════════════════════════════════
# UI Layout
# ═══════════════════════════════════════════════════════════════════════
def build_ui():
    dpg.create_context()

    with dpg.window(label="Chaos Synth — Granular", width=920, height=620,
                    tag="main_window"):
        # ── PARTICLE ──
        dpg.add_text("PARTICLE", color=(255, 200, 100))
        with dpg.group(horizontal=True):
            _add_slider("rate", "Rate", 1, 200, 20, "%.0f")
            _add_slider("pitch", "Pitch", 20, 8000, 440, "%.0f Hz")
            _add_slider("size", "Size", 0.003, 0.5, 0.05, "%.3f s")
            _add_slider("feedback", "Feedback", 0, 1, 0.3, "%.2f")

        dpg.add_spacer(height=8)

        # ── SPREAD ──
        dpg.add_text("SPREAD", color=(100, 200, 255))
        with dpg.group(horizontal=True):
            _add_slider("pitch_spread", "Pitch ±", 0, 2, 0.1, "%.1f oct")
            _add_slider("pan_spread", "Pan", 0, 1, 0.5, "%.2f")
            _add_slider("position_spread", "Pos ±", 0, 1, 0.1, "%.2f")

        dpg.add_spacer(height=8)

        # ── TIMBRE MAP ──
        dpg.add_text("TIMBRE MAP", color=(200, 150, 255))
        with dpg.group(horizontal=True):
            dpg.add_combo(tag="timbre_map_combo", label="Map",
                          items=_timbre_map.map_names, default_value="Gradual",
                          callback=_on_map_change, width=130)
            dpg.add_slider_float(tag="position_slider", label="Position",
                                 default_value=0.5, min_value=0.0, max_value=1.0,
                                 callback=_on_param_change, user_data="position",
                                 width=300, height=60, vertical=True)

        dpg.add_spacer(height=8)

        # ── LFOs ──
        dpg.add_text("LFO", color=(150, 255, 150))
        for i in range(3):
            with dpg.group(horizontal=True):
                dpg.add_text(f"LFO{i+1}")
                dpg.add_combo(tag=f"lfo{i}_target", items=LFO_TARGETS,
                              default_value=_params[f'lfo{i}_target'], width=80,
                              callback=_on_lfo_change, user_data=f'lfo{i}_target')
                dpg.add_combo(tag=f"lfo{i}_wave", items=LFO_WAVES,
                              default_value=_params[f'lfo{i}_wave'], width=80,
                              callback=_on_lfo_change, user_data=f'lfo{i}_wave')
                dpg.add_slider_float(tag=f"lfo{i}_rate", label="Rate",
                                     default_value=_params[f'lfo{i}_rate'],
                                     min_value=0.05, max_value=20.0,
                                     callback=_on_lfo_change, user_data=f'lfo{i}_rate',
                                     width=80, height=40, vertical=True)
                dpg.add_slider_float(tag=f"lfo{i}_depth", label="Depth",
                                     default_value=_params[f'lfo{i}_depth'],
                                     min_value=0.0, max_value=1.0,
                                     callback=_on_lfo_change, user_data=f'lfo{i}_depth',
                                     width=80, height=40, vertical=True)

    dpg.create_viewport(title="Chaos Synth — Granular", width=940, height=660)
    dpg.setup_dearpygui()
    dpg.show_viewport()

def _add_slider(key, label, vmin, vmax, default, fmt):
    with dpg.group():
        dpg.add_text(label)
        dpg.add_slider_float(tag=key, label="",
                             default_value=default, min_value=vmin, max_value=vmax,
                             callback=_on_param_change, user_data=key,
                             width=100, height=80, vertical=True, format=fmt)

def _on_lfo_change(sender, app_data, user_data):
    with _params_lock:
        _params[user_data] = app_data

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    build_ui()
    _start_audio()
    try:
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
    finally:
        _stop_audio()
        dpg.destroy_context()

if __name__ == '__main__':
    main()
