#!/usr/bin/env python3
"""chaos-synth — Granular Synthesis Engine (Phase 4b)

3D timbre space: Pos E→exciter, Pos B→body, Pos M→modulator.
MIDI pitch, 6 LFOs (3 dedicated position + 3 general purpose).

Run: python src/ui/app.py
"""

import sys, os, threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from main import (
    SAMPLE_RATE, BLOCK_SIZE, LFO,
    VoicePool, SelfSampleBuffer, DelayNetwork,
    EXCITERS, BODIES,
)

import dearpygui.dearpygui as dpg
import sounddevice as sd

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
def midi_to_hz(note):
    """MIDI note number → Hz. 69 = A4 = 440Hz."""
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))

# ═══════════════════════════════════════════════════════════════════════
# Shared State
# ═══════════════════════════════════════════════════════════════════════
_params_lock = threading.Lock()
_params = {
    # Particle
    'rate': 20.0,
    'pitch': 60,          # MIDI note (C4)
    'size': 0.1,          # seconds
    'feedback': 0.3,
    # Position (3D timbre)
    'pos_e': 0.5,         # exciter axis
    'pos_b': 0.5,         # body axis
    'pos_m': 0.5,         # modulator axis
    'pos_spread': 0.1,
    # Macros (E/B/M)
    'bite': 0.5, 'color': 0.5,
    'ring': 0.5, 'body_size': 0.5,
    'warp': 0.5, 'grit': 0.5,
    # Spread
    'pitch_spread': 2.0,  # semitones
    'pan_spread': 0.5,
    # POS LFOs (dedicated to position axes)
    'poslfo_e_wave': 'Sine', 'poslfo_e_rate': 0.3, 'poslfo_e_depth': 0.0,
    'poslfo_b_wave': 'Sine', 'poslfo_b_rate': 0.4, 'poslfo_b_depth': 0.0,
    'poslfo_m_wave': 'Sine', 'poslfo_m_rate': 0.5, 'poslfo_m_depth': 0.0,
    # General LFOs
    'lfo0_target': 'Pitch', 'lfo0_wave': 'Sine', 'lfo0_rate': 0.5, 'lfo0_depth': 0.0,
    'lfo1_target': 'Size',  'lfo1_wave': 'Sine', 'lfo1_rate': 0.3, 'lfo1_depth': 0.0,
    'lfo2_target': 'Pan',   'lfo2_wave': 'Sine', 'lfo2_rate': 0.2, 'lfo2_depth': 0.0,
}

# Engine instances
_pool = VoicePool(capacity=128, max_active=40)
_ssb = SelfSampleBuffer(duration_s=2.0)
_delay_net = DelayNetwork()
_pos_lfos = [LFO("PE"), LFO("PB"), LFO("PM")]
_lfos = [LFO("LFO1"), LFO("LFO2"), LFO("LFO3")]

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

    # ── Sync + tick POS LFOs (fixed targets: pos_e, pos_b, pos_m) ─────
    pos_lfo_vals = []
    for i, lfo in enumerate(_pos_lfos):
        axis = ['e', 'b', 'm'][i]
        lfo.waveform = p[f'poslfo_{axis}_wave']
        lfo.rate = p[f'poslfo_{axis}_rate']
        lfo.depth = p[f'poslfo_{axis}_depth']
        lfo.tick(dt)
        pos_lfo_vals.append(lfo.value * lfo.depth if lfo.depth > 0.001 else 0.0)

    # ── Sync + tick general LFOs ───────────────────────────────────────
    for i, lfo in enumerate(_lfos):
        lfo.target = p[f'lfo{i}_target']
        lfo.waveform = p[f'lfo{i}_wave']
        lfo.rate = p[f'lfo{i}_rate']
        lfo.depth = p[f'lfo{i}_depth']
        lfo.tick(dt)

    # ── Apply general LFOs to parameters ───────────────────────────────
    def lfo_val(target):
        for lfo in _lfos:
            if lfo.target == target and lfo.depth > 0.001:
                return lfo.value * lfo.depth
        return 0.0

    rate = max(1.0, p['rate'] * (1.0 + lfo_val('Rate')))
    pitch_midi = max(21.0, min(108.0, p['pitch'] + lfo_val('Pitch') * 24.0))
    size = max(0.001, min(2.0, p['size'] * (1.0 + lfo_val('Size'))))
    pan = np.clip(p['pan_spread'] + lfo_val('Pan'), 0.0, 1.0)

    # 3D position with dedicated LFOs
    pos_e = np.clip(p['pos_e'] + pos_lfo_vals[0] * 0.5, 0.0, 1.0)
    pos_b = np.clip(p['pos_b'] + pos_lfo_vals[1] * 0.5, 0.0, 1.0)
    pos_m = np.clip(p['pos_m'] + pos_lfo_vals[2] * 0.5, 0.0, 1.0)

    # ── Grains this block ──────────────────────────────────────────────
    _cb_state['accumulator'] += rate * dt
    n_grains = int(_cb_state['accumulator'])
    _cb_state['accumulator'] -= n_grains

    # ── Trigger grains ─────────────────────────────────────────────────
    spread = p['pos_spread']
    for _ in range(n_grains):
        # 3D position → combo
        eid = int(np.clip(pos_e + np.random.uniform(-spread, spread), 0.0, 0.999) * 12)
        bid = int(np.clip(pos_b + np.random.uniform(-spread, spread), 0.0, 0.999) * 10)
        mid = int(np.clip(pos_m + np.random.uniform(-spread, spread), 0.0, 0.999) * 7)

        # Pitch: MIDI + spread (semitones)
        note = pitch_midi + np.random.uniform(-p['pitch_spread'], p['pitch_spread'])
        pfreq = midi_to_hz(np.clip(note, 21.0, 120.0))

        # Size in samples
        psize_samples = max(4, int(size * SAMPLE_RATE))

        # Amplitude
        amp = 0.3 / max(1.0, n_grains ** 0.5)

        # Trigger voice
        _pool.trigger(eid, bid, mid, pfreq, amp, 0.5,
                       bite=p['bite'], color=p['color'],
                       ring=p['ring'], body_size=p['body_size'],
                       warp=p['warp'], grit=p['grit'])

        # Override grain: truncate + self-sample mix
        for v in _pool.voices:
            if v['active'] and v['age'] == 0:
                grain = v['buffer']
                g_slice = grain[:psize_samples]
                # If requested size > grain length, extend with decay tail
                if psize_samples > len(g_slice):
                    tail_len = psize_samples - len(g_slice)
                    last_val = g_slice[-1] if len(g_slice) > 0 else 0.0
                    decay_tau = int(SAMPLE_RATE * 0.02)  # 20ms decay
                    decay = np.exp(-np.arange(tail_len) / decay_tau).astype(np.float32)
                    tail = last_val * decay
                    g_slice = np.concatenate([g_slice, tail])
                if p['feedback'] > 0.001:
                    ssb_grain = _ssb.snatch(length=len(g_slice))
                    mix = min(len(g_slice), len(ssb_grain))
                    g_slice = g_slice[:mix] * (1.0 - p['feedback'] * 0.6) + ssb_grain[:mix] * p['feedback'] * 0.6
                # Cosine fade-in/fade-out to prevent clicks at grain boundaries
                fade_len = min(16, len(g_slice))
                if fade_len > 0:
                    fade_in = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_len) / fade_len)
                    fade_out = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_len)[::-1] / fade_len)
                    g_slice[:fade_len] *= fade_in.astype(np.float32)
                    g_slice[-fade_len:] *= fade_out.astype(np.float32)
                # Ensure buffer is large enough (grain may have been extended)
                if len(g_slice) > len(v['buffer']):
                    v['buffer'] = np.zeros(len(g_slice), dtype=np.float32)
                v['buffer'][:len(g_slice)] = g_slice
                if len(g_slice) < len(v['buffer']):
                    v['buffer'][len(g_slice):] = 0.0
                v['duration'] = len(g_slice)
                # Per-voice pan
                v['pan'] = 0.5 + np.random.uniform(-1.0, 1.0) * pan * 0.5
                break

    # ── Render pool + effects ──────────────────────────────────────────
    _pool.render(outdata.T)
    _ssb.write(outdata.T.copy())
    _delay_net.set_feedback(p['feedback'])
    _delay_net.wet_mix = 0.05 + p['feedback'] * 0.7
    outdata[:] = _delay_net.process(outdata.T).T

# ═══════════════════════════════════════════════════════════════════════
# Audio Stream
# ═══════════════════════════════════════════════════════════════════════
def _start_audio():
    global _audio_stream
    _audio_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
        channels=2, dtype='float32', callback=_audio_callback)
    _audio_stream.start()
    print("Audio started.")

def _stop_audio():
    global _audio_stream
    if _audio_stream:
        _audio_stream.stop(); _audio_stream.close(); _audio_stream = None

# ═══════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════
def _on_change(sender, app_data, user_data):
    with _params_lock:
        _params[user_data] = app_data

LFO_TARGETS = ['Pitch', 'Size', 'Rate', 'Pan', 'Pitch Spread', 'Pos Spread']
LFO_WAVES = LFO.WAVEFORMS

def build_ui():
    dpg.create_context()

    with dpg.window(label="Chaos Synth — Granular", width=960, height=700):

        # ── PARTICLE ──
        dpg.add_text("PARTICLE", color=(255, 200, 100))
        with dpg.group(horizontal=True):
            _slider("rate", "Rate", 1, 200, 20, "%.0f/s")
            _slider("pitch", "Pitch", 21, 108, 60, "%.0f")
            _slider("size", "Size", 0.001, 2.0, 0.1, "%.3fs")
            _slider("feedback", "Feedback", 0, 1, 0.3, "%.2f")

        dpg.add_spacer(height=6)

        # ── POSITION (3D) ──
        dpg.add_text("POSITION  (E=exciter  B=body  M=modulator)", color=(200, 150, 255))
        with dpg.group(horizontal=True):
            _slider("pos_e", "Pos E", 0, 1, 0.5, "%.3f")
            _slider("pos_b", "Pos B", 0, 1, 0.5, "%.3f")
            _slider("pos_m", "Pos M", 0, 1, 0.5, "%.3f")
            _slider("pos_spread", "Spread", 0, 1, 0.1, "%.2f")

        dpg.add_spacer(height=6)

        # ── SPREAD ──
        dpg.add_text("SPREAD", color=(100, 200, 255))
        with dpg.group(horizontal=True):
            _slider("pitch_spread", "Pitch +-", 0, 24, 2, "%.0f sem")
            _slider("pan_spread", "Pan", 0, 1, 0.5, "%.2f")

        dpg.add_spacer(height=6)

        # ── MACRO ──
        dpg.add_text("MACRO  (E: Bite/Color  B: Ring/Body  M: Warp/Grit)", color=(255, 180, 100))
        with dpg.group(horizontal=True):
            _slider("bite", "Bite", 0, 1, 0.5, "%.2f")
            _slider("color", "Color", 0, 1, 0.5, "%.2f")
            _slider("ring", "Ring", 0, 1, 0.5, "%.2f")
            _slider("body_size", "Body", 0, 1, 0.5, "%.2f")
            _slider("warp", "Warp", 0, 1, 0.5, "%.2f")
            _slider("grit", "Grit", 0, 1, 0.5, "%.2f")

        dpg.add_spacer(height=6)

        # ── POS LFOs ──
        dpg.add_text("POS LFO  (E / B / M)", color=(150, 255, 150))
        for i, axis in enumerate(['e', 'b', 'm']):
            with dpg.group(horizontal=True):
                dpg.add_text(f"  {axis.upper()}:")
                dpg.add_combo(tag=f"poslfo_{axis}_wave", items=LFO_WAVES,
                              default_value="Sine", width=75,
                              callback=_on_change, user_data=f"poslfo_{axis}_wave")
                _lfo_slider(f"poslfo_{axis}_rate", 0.05, 10, 0.3)
                _lfo_slider(f"poslfo_{axis}_depth", 0, 1, 0.0)

        dpg.add_spacer(height=6)

        # ── General LFOs ──
        dpg.add_text("LFO", color=(150, 255, 150))
        for i in range(3):
            with dpg.group(horizontal=True):
                dpg.add_text(f"LFO{i+1}")
                dpg.add_combo(tag=f"lfo{i}_target", items=LFO_TARGETS,
                              default_value=_params[f'lfo{i}_target'], width=90,
                              callback=_on_change, user_data=f'lfo{i}_target')
                dpg.add_combo(tag=f"lfo{i}_wave", items=LFO_WAVES,
                              default_value="Sine", width=75,
                              callback=_on_change, user_data=f'lfo{i}_wave')
                _lfo_slider(f"lfo{i}_rate", 0.05, 20, _params[f'lfo{i}_rate'])
                _lfo_slider(f"lfo{i}_depth", 0, 1, _params[f'lfo{i}_depth'])

        dpg.add_spacer(height=6)

    dpg.create_viewport(title="Chaos Synth — Granular", width=980, height=730)
    dpg.setup_dearpygui()
    dpg.show_viewport()

def _slider(key, label, vmin, vmax, default, fmt):
    with dpg.group():
        dpg.add_text(label)
        dpg.add_slider_float(tag=key, label="",
                             default_value=default, min_value=vmin, max_value=vmax,
                             callback=_on_change, user_data=key,
                             width=100, height=70, vertical=True, format=fmt)

def _lfo_slider(tag, vmin, vmax, default):
    dpg.add_slider_float(tag=tag, label="",
                         default_value=default, min_value=vmin, max_value=vmax,
                         callback=_on_change, user_data=tag,
                         width=65, height=35, vertical=True)

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
