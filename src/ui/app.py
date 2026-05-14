#!/usr/bin/env python3
"""chaos-synth — Real-time Performance UI (Phase 3)

DearPyGui interface with 9 macro controls, real-time spectrum display,
Lorenz attractor 2D trajectory visualization, preset save/load, and
full audio engine integration via sounddevice callback.

Run: python src/ui/app.py
"""

import sys, os, threading, queue
import numpy as np

# ── Import engine from parent package ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from main import (
    SAMPLE_RATE, BLOCK_SIZE, LorenzAttractor, ManifoldMapper,
    VoicePool, CouplingField, CellularAutomaton, LSystem,
    DelayNetwork, SelfSampleBuffer, LongTermFeedback,
    map_tonic_spread_to_freq, map_dynamic_to_amp,
    compute_spectral_centroid, EXCITERS, BODIES, MODULATORS,
    exciter_transient,
)
from presets import PresetManager

import dearpygui.dearpygui as dpg
import sounddevice as sd


# ═══════════════════════════════════════════════════════════════════════
# Shared State (UI thread writes, audio thread reads)
# ═══════════════════════════════════════════════════════════════════════
_params_lock = threading.Lock()
_params = {
    'tonic': 0.5,
    'scale': 'Pentatonic',
    'dynamic': 0.5,
    'pitch_spread': 0.3,
    'material': 0.5,
    'density': 0.5,
    'mutation': 0.3,
    'coherence': 0.7,
    'feedback': 0.5,
}

_spectrum_queue = queue.Queue(maxsize=5)
_attractor_queue = queue.Queue(maxsize=10)

# ── Audio Engine State (owned by audio thread, UI may read) ───────────
_chaos = LorenzAttractor()
_manifold = ManifoldMapper(n_centroids=16)
_pool = VoicePool(capacity=128, max_active=16)
_coupling = CouplingField()
_ca = CellularAutomaton(width=64, rule=30)
_lsys = LSystem(axiom="ABCB", rules={"A": "ABC", "B": "BAB", "C": "CA"})
_ssb = SelfSampleBuffer(duration_s=2.0)
_delay_net = DelayNetwork()
_ltfb = LongTermFeedback()

# Callback-local mutable state (accessed only from audio thread)
_cb_state = {
    'ca_counter': 0,
    'lsys_counter': 0,
    'ltfb_frame': 0,
    'trigger_gates': [True] * 8,
    'note_counter': 0,        # frames since last note change
    'held_freq': 440.0,       # currently held pitch (Hz)
    'target_freq': 440.0,     # portamento target pitch (Hz)
    'portamento_counter': 0,  # frames into current glide
    'portamento_duration': 0, # total glide frames (coherence * 30)
    'feedback_state': {
        'centroid_history': [],
        'flux_history': [],
        'zcr_history': [],
        'silence_counter': 0,
        'sigma_drift': 0.0,
    },
}

_audio_stream = None


# ═══════════════════════════════════════════════════════════════════════
# Audio Callback (runs in sounddevice C thread)
# ═══════════════════════════════════════════════════════════════════════
def _audio_callback(outdata, frames, time_info, status):
    if status:
        print(f"audio callback status: {status}", flush=True)
    outdata.fill(0.0)

    # Snapshot params under lock
    with _params_lock:
        p = dict(_params)

    tonic = p['tonic']
    scale = p['scale']
    dynamic = p['dynamic']
    pitch_spread = p['pitch_spread']
    material = p['material']
    density = p['density']
    mutation = p['mutation']
    coherence = p['coherence']
    feedback = p['feedback']

    st = _cb_state
    fb = st['feedback_state']

    # ── 0. Generative layer: CA rhythm + L-System melody ──────────────
    st['ca_counter'] += 1
    if st['ca_counter'] % 4 == 0:
        _ca.step()

        # Mutation: randomly change CA rule
        if np.random.rand() < mutation * 0.015:
            _ca.rule = int(np.random.choice([30, 90, 110, 150, 54, 184]))

        # Density: more gates open at high density
        ca_row = _ca.trigger_pattern()
        threshold = 1.0 - density  # 0 = all gates open, 1 = sparse
        # Gate is "open" if at least threshold fraction of cells are alive
        st['trigger_gates'] = ca_row

    st['lsys_counter'] += 1
    if st['lsys_counter'] % 20 == 0:
        _lsys.iterate()
        _ca.step()
    melody_notes = _lsys.to_notes(base_freq=110.0 * (0.5 + tonic)) if st['lsys_counter'] % 20 == 0 else []

    # ── 1. Step chaos engine → 3D state ───────────────────────────────
    state = _chaos.step()
    eid, bid, mid = _manifold.find_nearest(state)

    # Material: bias body selection (0=transient, 1=resonant)
    TRANSIENT_BODIES = {0, 2, 4, 5}   # dry, comb, nonlinear, freeze
    RESONANT_BODIES  = {1, 3, 6, 7, 8, 9}  # modal, allpass, waveguide, saturation, blur, pshift
    switch_prob = abs(material - 0.5) * 2.0  # 0 at center, 1 at extremes
    if np.random.rand() < switch_prob:
        if material > 0.5:
            # Bias toward resonant
            if bid in TRANSIENT_BODIES:
                bid = int(np.random.choice(list(RESONANT_BODIES)))
        else:
            # Bias toward transient
            if bid in RESONANT_BODIES:
                bid = int(np.random.choice(list(TRANSIENT_BODIES)))

    # ── 2. Map to frequency & amplitude ───────────────────────────────
    gate_idx = int(state[2] * 8) % 8

    # Note-hold: mutation controls how often frequency changes
    # Coherence controls portamento (log-domain glide) between notes
    N = max(1, int((1 - mutation) ** 3 * 140))
    if st['note_counter'] >= N:
        # New note: recompute frequency from current chaos state
        new_freq = map_tonic_spread_to_freq(state, tonic, scale, pitch_spread)
        # L-system melody override only at note boundaries
        if melody_notes and np.random.rand() < 0.2:
            new_freq = melody_notes[np.random.randint(0, len(melody_notes))]
        # Portamento setup: glide from current held_freq → target_freq
        # coherence=0 → duration=0 (hard cut), coherence=1 → 30 frames (~174ms)
        st['target_freq'] = new_freq
        st['portamento_duration'] = int(coherence * 30)
        st['portamento_counter'] = 0
        st['note_counter'] = 0
    # Portamento glide: logarithmic interpolation for perceptually smooth pitch slide
    if st['portamento_counter'] < st['portamento_duration']:
        t = st['portamento_counter'] / st['portamento_duration']
        log_old = np.log2(st['held_freq'])
        log_new = np.log2(st['target_freq'])
        st['held_freq'] = float(2 ** (log_old + (log_new - log_old) * t))
        st['portamento_counter'] += 1
    else:
        # Glide complete (or duration=0), snap to target
        st['held_freq'] = st['target_freq']
    freq = st['held_freq']
    st['note_counter'] += 1

    amp = map_dynamic_to_amp(state, dynamic)

    # Density: reduce gate penalty
    if not st['trigger_gates'][gate_idx]:
        amp *= max(0.1, 1.0 - density)  # high density → less penalty

    # ── 3. Trigger voice (handle exciter #11 transient self-sample) ───
    if eid == 11:
        # Self-sample transient: pass snatch from ring buffer
        grain = BODIES.get(bid, lambda x, f: x)(
            exciter_transient(freq, SAMPLE_RATE, _ssb.snatch()),
            freq, SAMPLE_RATE)
    else:
        efn = EXCITERS.get(eid, EXCITERS[0])
        bfn = BODIES.get(bid, BODIES[0])
        grain = bfn(efn(freq, SAMPLE_RATE), freq, SAMPLE_RATE)

    # Feedback: mix self-sample buffer into main voice (after body, before modulator)
    ssb_length = min(256, len(grain))
    ssb_grain = _ssb.snatch(length=ssb_length)
    if len(ssb_grain) != len(grain):
        min_len = min(len(grain), len(ssb_grain))
        grain = grain[:min_len] * (1.0 - feedback * 0.6) + ssb_grain[:min_len] * feedback * 0.6
    else:
        grain = grain * (1.0 - feedback * 0.6) + ssb_grain * feedback * 0.6

    # Apply modulator
    if mid == 1:
        grain = _mod_tremolo(grain)
    elif mid == 2:
        grain = _mod_vibrato(grain)
    elif mid == 3:
        grain = _mod_phase_dist(grain)
    elif mid == 4:
        grain = _mod_ringmod(grain)
    elif mid == 5:
        grain = _mod_bitcrush(grain)
    elif mid == 6:
        grain = _mod_stereo_width(grain)
    # else static (passthrough)

    grain = np.clip(grain * amp, -1.0, 1.0).astype(np.float32)

    # Manually insert grain into voice pool (replicate trigger logic)
    _pool.trigger(eid, bid, mid, freq, amp, float(state[2]))
    # Override the buffer in the voice slot with our material-biased grain
    for v in _pool.voices:
        if v['active'] and v['age'] == 0:
            v['buffer'][:len(grain)] = grain[:len(v['buffer'])]

    # Density: trigger extra voices
    extra_count = int(density * 6)
    for _ in range(extra_count):
        extra_state = _chaos.step()
        eeid, ebid, emid = _manifold.find_nearest(extra_state)
        eamp = map_dynamic_to_amp(extra_state, dynamic) * 0.4
        # Use the same held frequency for extra voices
        efreq = freq
        if not st['trigger_gates'][int(extra_state[2] * 8) % 8]:
            eamp *= max(0.05, 1.0 - density)

        if eeid == 11:
            egrain = BODIES.get(ebid, lambda x, f: x)(
                exciter_transient(efreq, SAMPLE_RATE, _ssb.snatch()),
                efreq, SAMPLE_RATE)
        else:
            efn2 = EXCITERS.get(eeid, EXCITERS[0])
            bfn2 = BODIES.get(ebid, BODIES[0])
            egrain = bfn2(efn2(efreq, SAMPLE_RATE), efreq, SAMPLE_RATE)

        # Feedback: mix self-sample buffer into extra voice (after body, before modulator)
        ssb_len2 = min(256, len(egrain))
        ssb_grain2 = _ssb.snatch(length=ssb_len2)
        if len(ssb_grain2) != len(egrain):
            min_len2 = min(len(egrain), len(ssb_grain2))
            egrain = egrain[:min_len2] * (1.0 - feedback * 0.6) + ssb_grain2[:min_len2] * feedback * 0.6
        else:
            egrain = egrain * (1.0 - feedback * 0.6) + ssb_grain2 * feedback * 0.6

        # Apply modulator
        if emid == 1:
            egrain = _mod_tremolo(egrain)
        elif emid == 2:
            egrain = _mod_vibrato(egrain)
        elif emid == 3:
            egrain = _mod_phase_dist(egrain)
        elif emid == 4:
            egrain = _mod_ringmod(egrain)
        elif emid == 5:
            egrain = _mod_bitcrush(egrain)
        elif emid == 6:
            egrain = _mod_stereo_width(egrain)

        egrain = np.clip(egrain * eamp, -1.0, 1.0).astype(np.float32)

        _pool.trigger(eeid, ebid, emid, efreq, eamp, float(extra_state[2]))
        for v in _pool.voices:
            if v['active'] and v['age'] == 0:
                v['buffer'][:len(egrain)] = egrain[:len(v['buffer'])]

    # ── 4. Render voice pool ──────────────────────────────────────────
    _pool.render(outdata.T)

    # ── 5. Self-sample buffer write ───────────────────────────────────
    _ssb.write(outdata.T.copy())

    # ── 6. Feature extraction ─────────────────────────────────────────
    mono = outdata.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    centroid = compute_spectral_centroid(mono, SAMPLE_RATE)
    zcr = float(np.sum(np.abs(np.diff(np.sign(mono)))) / (2.0 * len(mono))) if len(mono) > 0 else 0.0
    prev_cent = fb['centroid_history'][-1] if fb['centroid_history'] else centroid
    flux = abs(centroid - prev_cent)

    # Push spectrum to UI queue
    try:
        spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)).astype(np.float32)))
        _spectrum_queue.put_nowait(spec[:128].astype(np.float32).copy())
    except queue.Full:
        pass

    # Push attractor state to UI queue
    try:
        _attractor_queue.put_nowait((float(_chaos.x), float(_chaos.y)))
    except queue.Full:
        pass

    # ── 7. Long-term feedback ─────────────────────────────────────────
    _ltfb.feed(centroid, flux, rms)
    st['ltfb_frame'] += 1
    if st['ltfb_frame'] >= _ltfb.window_size:
        st['ltfb_frame'] = 0
        lt_mod = _ltfb.tick()
        if lt_mod and hasattr(_chaos, 'rho'):
            target_rho = 15.0 + lt_mod.get('lt_brightness', 0.5) * 25.0
            _chaos.rho += (target_rho - _chaos.rho) * 0.02
        if lt_mod and hasattr(_chaos, 'dt'):
            activity = lt_mod.get('lt_activity', 0.5)
            target_dt = 0.005 + activity * 0.04
            _chaos.dt += (target_dt - _chaos.dt) * 0.02

    # ── 8. Smooth histories ───────────────────────────────────────────
    fb['centroid_history'].append(centroid)
    fb['flux_history'].append(flux)
    fb['zcr_history'].append(zcr)
    for h in ['centroid_history', 'flux_history', 'zcr_history']:
        if len(fb[h]) > 10:
            fb[h].pop(0)

    avg_centroid = np.mean(fb['centroid_history']) if fb['centroid_history'] else centroid
    avg_flux = np.mean(fb['flux_history']) if fb['flux_history'] else flux
    avg_zcr = np.mean(fb['zcr_history']) if fb['zcr_history'] else zcr

    # ── 9. Feedback: coherence affects sigma range ────────────────────
    if hasattr(_chaos, 'sigma'):
        sigma_min = 8.0 - coherence * 4.0    # 4.0 → 8.0
        sigma_max = 11.0 + (1.0 - coherence) * 14.0  # 11.0 → 25.0
        target_sigma = sigma_min + (1.0 - min(avg_centroid / 5000.0, 1.0)) * (sigma_max - sigma_min)
        _chaos.sigma += (target_sigma - _chaos.sigma) * 0.01

    # ── 10. Feedback: spectral flux → dt ──────────────────────────────
    if hasattr(_chaos, 'dt'):
        dt_min = 0.003 + coherence * 0.012  # 0.003 → 0.015
        dt_max = dt_min + (1.0 - coherence) * 0.04  # wider range at low coherence
        target_dt = dt_min + (1.0 - min(avg_flux / 1000.0, 1.0)) * (dt_max - dt_min)
        _chaos.dt += (target_dt - _chaos.dt) * 0.01

    # ── 11. Feedback: ZCR → voice count ───────────────────────────────
    target_active = int(4 + (1.0 - min(avg_zcr, 1.0)) * 20)
    _pool.max_active = max(4, min(32, target_active))

    # ── 12. Coupling field ────────────────────────────────────────────
    coupling_gain = 0.1 + feedback * 0.6
    _coupling.deposit(rms * 2.0)
    _coupling.tick()
    extra = _coupling.read()
    outdata[:] *= (1.0 + extra * coupling_gain)

    # ── 13. Delay network (feedback-controlled wet mix + cross-feedback) ──
    _delay_net.set_feedback(feedback)
    _delay_net.wet_mix = 0.05 + feedback * 0.7
    outdata[:] = _delay_net.process(outdata.T).T

    # ── 14. Cold start noise injection ────────────────────────────────
    fb['silence_counter'] += frames
    if rms > 1e-4:
        fb['silence_counter'] = 0
    if fb['silence_counter'] > int(SAMPLE_RATE * 0.5):
        noise = np.random.randn(frames).astype(np.float32) * 0.01
        outdata[:, 0] += noise
        outdata[:, 1] += noise
        fb['silence_counter'] = 0

    # ── 15. Sigma drift (amplified by mutation) ───────────────────────
    if hasattr(_chaos, 'sigma'):
        fb['sigma_drift'] += np.random.randn() * (0.001 + mutation * 0.006)
        fb['sigma_drift'] = np.clip(fb['sigma_drift'], -1.5, 1.5)
        drift = fb['sigma_drift']
        sigma_range = 14.0 * (1.0 - coherence * 0.7)
        _chaos.sigma = np.clip(10.0 + drift * sigma_range, 3.0, 28.0)


# ── Modulator helpers (avoid closures in audio callback) ──────────────
def _mod_tremolo(buf):
    t = np.arange(len(buf), dtype=np.float32) / SAMPLE_RATE
    return (buf * (0.5 + 0.5 * np.sin(2.0 * np.pi * 5.0 * t))).astype(np.float32)

def _mod_vibrato(buf):
    n = len(buf)
    t = np.arange(n, dtype=np.float32)
    mod = 1.0 + 0.003 * np.sin(2.0 * np.pi * 5.0 * t / SAMPLE_RATE * n)
    idx = np.clip((np.arange(n, dtype=np.float32) * mod).astype(np.int32), 0, n - 1)
    return buf[idx].astype(np.float32)

def _mod_phase_dist(buf):
    n = len(buf)
    phase = np.arange(n, dtype=np.float32) / float(n)
    bent = phase + 0.5 * np.sin(2.0 * np.pi * phase)
    bent = bent / bent.max() * (n - 1)
    idx = bent.astype(np.int32)
    frac = bent - idx
    idx = np.clip(idx, 0, n - 2)
    return (buf[idx] * (1.0 - frac) + buf[idx + 1] * frac).astype(np.float32)

def _mod_ringmod(buf):
    t = np.arange(len(buf), dtype=np.float32) / SAMPLE_RATE
    return (buf * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)

def _mod_bitcrush(buf):
    levels = 2.0 ** 6
    return (np.round(buf * levels) / levels).astype(np.float32)

def _mod_stereo_width(buf):
    # Not used in mono rendering path, passthrough
    return buf.astype(np.float32)


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
    """Slider/combo changed → update shared params dict."""
    key = user_data
    with _params_lock:
        _params[key] = app_data


def _on_save_preset():
    """Save current state as a preset."""
    name = dpg.get_value("preset_name_input")
    if not name or not name.strip():
        print("Preset name is empty.")
        return
    name = name.strip()
    with _params_lock:
        macros = dict(_params)
    preset = PresetManager.capture(
        _chaos, _manifold, _pool, _coupling, _delay_net, macros
    )
    PresetManager.save(preset, name)
    print(f"Preset '{name}' saved.")
    _refresh_preset_list()


def _on_load_preset():
    """Load selected preset and apply to shared params."""
    preset_name = dpg.get_value("preset_list_combo")
    if not preset_name:
        print("No preset selected.")
        return
    preset = PresetManager.load(preset_name)
    macros = preset.get('macros', {})
    with _params_lock:
        for k in _params:
            if k in macros:
                _params[k] = macros[k]
    # Apply to UI sliders
    _sync_ui_from_params()
    print(f"Preset '{preset_name}' loaded.")


def _sync_ui_from_params():
    """Read shared params and update all UI controls."""
    with _params_lock:
        p = dict(_params)
    for key, tag in _slider_tags.items():
        dpg.set_value(tag, p[key])
    dpg.set_value("scale_combo", p['scale'])


def _refresh_preset_list():
    """Refresh the preset combo with available presets."""
    presets = PresetManager.list_presets()
    dpg.configure_item("preset_list_combo", items=presets)
    if presets:
        dpg.set_value("preset_list_combo", presets[0])


# ── Tags for sliders (key → dpg tag) ──────────────────────────────────
_slider_tags = {}


# ═══════════════════════════════════════════════════════════════════════
# UI Construction
# ═══════════════════════════════════════════════════════════════════════
def _build_ui():
    """Build the complete DearPyGui UI."""
    dpg.create_context()

    # ── Window ────────────────────────────────────────────────────────
    with dpg.window(
        label="Chaos Synth — Real-time Performance UI",
        width=900, height=700,
        no_close=True,
        tag="main_window",
    ):
        # ── PERFORMANCE section ───────────────────────────────────────
        dpg.add_spacer(height=4)
        dpg.add_text("PERFORMANCE", color=(255, 220, 80, 255))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            # Tonic slider: vertical
            with dpg.group():
                dpg.add_text("Tonic")
                tag = dpg.add_slider_float(
                    label="", default_value=0.5, min_value=0.0, max_value=1.0,
                    width=100, height=120, vertical=True,
                    callback=_on_param_change, user_data='tonic',
                )
                _slider_tags['tonic'] = tag

            # Scale combo
            with dpg.group():
                dpg.add_text("Scale")
                dpg.add_combo(
                    items=["Chromatic", "Pentatonic", "Major", "Minor",
                           "Microtonal", "Harmonic"],
                    default_value="Pentatonic",
                    width=110,
                    callback=_on_param_change, user_data='scale',
                    tag="scale_combo",
                )

            # Dynamic slider: vertical
            with dpg.group():
                dpg.add_text("Dynamic")
                tag = dpg.add_slider_float(
                    label="", default_value=0.5, min_value=0.0, max_value=1.0,
                    width=100, height=120, vertical=True,
                    callback=_on_param_change, user_data='dynamic',
                )
                _slider_tags['dynamic'] = tag

            # Spread slider: vertical
            with dpg.group():
                dpg.add_text("Spread")
                tag = dpg.add_slider_float(
                    label="", default_value=0.3, min_value=0.0, max_value=1.0,
                    width=100, height=120, vertical=True,
                    callback=_on_param_change, user_data='pitch_spread',
                )
                _slider_tags['pitch_spread'] = tag

        dpg.add_spacer(height=6)

        # ── ECOSYSTEM section ─────────────────────────────────────────
        dpg.add_text("ECOSYSTEM", color=(80, 180, 255, 255))
        dpg.add_separator()

        with dpg.group(horizontal=True):
            eco_controls = [
                ('material', 'Material', 0.5),
                ('density', 'Density', 0.5),
                ('mutation', 'Mutation', 0.3),
                ('coherence', 'Coherence', 0.7),
                ('feedback', 'Feedback', 0.5),
            ]
            for key, label, default in eco_controls:
                with dpg.group():
                    dpg.add_text(label)
                    tag = dpg.add_slider_float(
                        label="", default_value=default, min_value=0.0, max_value=1.0,
                        width=100, height=120, vertical=True,
                        callback=_on_param_change, user_data=key,
                    )
                    _slider_tags[key] = tag

        dpg.add_spacer(height=6)

        # ── SPECTRUM plot ─────────────────────────────────────────────
        dpg.add_text("SPECTRUM", color=(140, 255, 140, 255))
        dpg.add_separator()
        with dpg.plot(label="", height=200, width=-1, tag="spectrum_plot",
                      no_menus=True, no_title=True):
            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency bin", tag="spec_x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Magnitude", tag="spec_y_axis")
            # Initialize with empty data
            x_init = list(range(128))
            y_init = [0.0] * 128
            dpg.add_line_series(
                x_init, y_init,
                parent="spec_y_axis", tag="spectrum_line",
            )
        dpg.add_spacer(height=4)

        # ── ATTRACTOR visualization ───────────────────────────────────
        dpg.add_text("LORENZ ATTRACTOR (XY projection)", color=(255, 140, 140, 255))
        dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.drawlist(width=870, height=150, tag="attractor_drawlist"):
                # Initial empty polyline
                dpg.draw_polyline(
                    [(0, 0), (1, 1)],
                    color=(255, 100, 100, 255),
                    thickness=1.5,
                    tag="attractor_polyline",
                )
            # Legend
            with dpg.group():
                dpg.add_text("X →")
                dpg.add_text("Y ↓")
                dpg.add_text("", tag="attractor_info")

        dpg.add_spacer(height=6)

        # ── PRESETS section ───────────────────────────────────────────
        dpg.add_text("PRESETS", color=(220, 180, 255, 255))
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                label="Name", default_value="my_preset",
                width=150, tag="preset_name_input",
            )
            dpg.add_button(label="Save", callback=_on_save_preset, width=60)
            dpg.add_button(label="Load", callback=_on_load_preset, width=60)
            presets = PresetManager.list_presets()
            dpg.add_combo(
                items=presets,
                default_value=presets[0] if presets else "",
                label="",
                width=150,
                tag="preset_list_combo",
            )

        dpg.add_spacer(height=4)

    # ── Theme / styling ───────────────────────────────────────────────
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (18, 18, 22, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (32, 32, 40, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (210, 210, 220, 255))
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (100, 200, 100, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
    dpg.bind_theme(global_theme)

    # ── Viewport ──────────────────────────────────────────────────────
    dpg.create_viewport(title="Chaos Synth v0.4.0", width=920, height=740)
    dpg.setup_dearpygui()
    dpg.show_viewport()


# ═══════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════
_attractor_history = []  # list of (x, y) tuples, max 200


def run_ui():
    """Build UI, start audio, enter render loop."""
    global _attractor_history

    _build_ui()
    _refresh_preset_list()
    _start_audio()

    print("chaos-synth UI v0.4.0 [Phase 3 — Real-time Performance UI]")
    print(f"  Chaos: {type(_chaos).__name__}")
    print(f"  Manifold: {len(_manifold.centroids)} centroids")
    print(f"  Pool: {_pool.capacity} slots, max {_pool.max_active} active")
    print(f"  S/R: {SAMPLE_RATE}Hz / block {BLOCK_SIZE}")
    print("  Close window to stop.")

    try:
        while dpg.is_dearpygui_running():
            # ── Update spectrum plot ──────────────────────────────────
            try:
                spec = _spectrum_queue.get_nowait()
                x_vals = list(range(len(spec)))
                y_vals = spec.tolist()
                dpg.set_value("spectrum_line", [x_vals, y_vals])
                # Auto-fit Y axis
                ymax = max(float(np.max(spec)), 1e-6)
                dpg.set_axis_limits("spec_y_axis", 0.0, ymax * 1.1)
                dpg.set_axis_limits("spec_x_axis", 0.0, float(len(spec)))
            except queue.Empty:
                pass

            # ── Update attractor visualization ────────────────────────
            try:
                while True:
                    pt = _attractor_queue.get_nowait()
                    _attractor_history.append(pt)
                    if len(_attractor_history) > 200:
                        _attractor_history.pop(0)
            except queue.Empty:
                pass

            if _attractor_history:
                # Map attractor [0,1]² to drawlist coordinates [0,870]×[0,150]
                # Flip Y for screen coordinates
                pts = [(int(x * 850) + 5, int((1.0 - y) * 140) + 5)
                       for x, y in _attractor_history]
                dpg.configure_item("attractor_polyline", points=pts)
                # Update info
                last = _attractor_history[-1]
                dpg.set_value("attractor_info",
                              f"x={last[0]:.3f}\ny={last[1]:.3f}\npts={len(_attractor_history)}")

            # ── Render one frame ──────────────────────────────────────
            dpg.render_dearpygui_frame()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        _stop_audio()
        dpg.destroy_context()


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    run_ui()
