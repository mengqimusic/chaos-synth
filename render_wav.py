#!/usr/bin/env python3
"""Offline render: Chaos Synth -> WAV file (Phase 3 — Full Integration)"""
import sys, os
os.chdir('/Users/mengwu/Documents/Code/chaos-synth')
sys.path.insert(0, 'src')

import numpy as np
from scipy.io import wavfile
from main import *

# Performance params
DURATION = 15  # longer to hear long-term feedback evolution
TONIC = 0.45
SCALE = "Minor"
PITCH_SPREAD = 0.6
DYNAMIC = 0.55

OUTPUT = "/Users/mengwu/Documents/Code/chaos-synth/output/phase3_render.wav"
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

chaos = LorenzAttractor()
manifold = ManifoldMapper(n_centroids=16)
pool = VoicePool(capacity=128, max_active=16)
coupling = CouplingField()
ca = CellularAutomaton(width=64, rule=30)
lsys = LSystem("ABCB", {"A": "ABC", "B": "BAB", "C": "CA"})

# Phase 3 components
ssb = SelfSampleBuffer(duration_s=2.0)
delay_net = DelayNetwork()
delay_net.wet_mix = 0.45  # more audible than default 0.3
ltfb = LongTermFeedback()
ltfb_frame_counter = 0

trigger_gates = [True] * 8
ca_counter = ls_counter = 0
fb = {'centroid_history': [], 'flux_history': [], 'zcr_history': [], 'silence_counter': 0}

total_frames = DURATION * SAMPLE_RATE
output = np.zeros((total_frames, 2), dtype=np.float32)
written = 0

print(f"Rendering {DURATION}s @ {SAMPLE_RATE}Hz...", flush=True)
print(f"  tonic={TONIC:.2f} scale={SCALE} spread={PITCH_SPREAD:.2f} dyn={DYNAMIC:.2f}", flush=True)

while written < total_frames:
    frames = min(BLOCK_SIZE, total_frames - written)
    out = np.zeros((frames, 2), dtype=np.float32)

    ca_counter += 1
    if ca_counter % 4 == 0:
        ca.step()
        trigger_gates = ca.trigger_pattern()

    ls_counter += 1
    if ls_counter % 20 == 0:
        lsys.iterate()
    melody_notes = lsys.to_notes(110.0) if ls_counter % 20 == 0 else []

    state = chaos.step()
    eid, bid, mid = manifold.find_nearest(state)
    # Phase 3: use performance handle mapping
    freq = map_tonic_spread_to_freq(state, TONIC, SCALE, PITCH_SPREAD)
    if melody_notes and np.random.random() < 0.2:
        freq = melody_notes[np.random.randint(0, len(melody_notes))]
    amp = map_dynamic_to_amp(state, DYNAMIC)
    if not trigger_gates[int(state[2]*8) % 8]:
        amp *= 0.3
    # Phase 3: pass self-sample buffer for exciter #11 (transient snatch)
    sample_buf = ssb.snatch() if eid == 11 else None
    pool.trigger(eid, bid, mid, freq, amp, float(state[2]), self_sample=sample_buf)

    buf = out.T.copy()
    pool.render(buf)
    out[:] = buf.T

    # Phase 3a: write to self-sample ring buffer (for exciter #11)
    ssb.write(out.T.copy())

    # Feature extraction + long-term feedback
    mono = out.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono**2)))
    centroid = compute_spectral_centroid(mono, SAMPLE_RATE)
    prev_centroid = fb['centroid_history'][-1] if fb['centroid_history'] else centroid
    flux = abs(centroid - prev_centroid)

    fb['centroid_history'].append(centroid)
    fb['flux_history'].append(flux)
    for h in ['centroid_history', 'flux_history']:
        if len(fb[h]) > 10:
            fb[h].pop(0)

    ltfb.feed(centroid, flux, rms)
    ltfb_frame_counter += 1
    if ltfb_frame_counter >= ltfb.window_size:
        ltfb_frame_counter = 0
        lt_mod = ltfb.tick()
        if lt_mod:
            if hasattr(chaos, 'rho'):
                target_rho = 15.0 + lt_mod.get('lt_brightness', 0.5) * 25.0
                chaos.rho += (target_rho - chaos.rho) * 0.02
            if hasattr(chaos, 'dt'):
                activity = lt_mod.get('lt_activity', 0.5)
                target_dt = 0.005 + activity * 0.04
                chaos.dt += (target_dt - chaos.dt) * 0.02

    # Short-term feedback
    if fb['centroid_history']:
        avg_centroid = np.mean(fb['centroid_history'])
        avg_flux = np.mean(fb['flux_history'])
        if hasattr(chaos, 'sigma'):
            target_sigma = 6.0 + (1.0 - min(avg_centroid / 5000.0, 1.0)) * 9.0
            chaos.sigma += (target_sigma - chaos.sigma) * 0.01
        if hasattr(chaos, 'dt'):
            target_dt_st = 0.005 + (1.0 - min(avg_flux / 1000.0, 1.0)) * 0.02
            chaos.dt += (target_dt_st - chaos.dt) * 0.005

    # Voices feedback
    if fb.get('zcr_history'):
        avg_zcr = np.mean(fb['zcr_history'])
        target_active = int(8 + (1.0 - min(avg_zcr, 1.0)) * 16)
        pool.max_active = max(4, min(32, target_active))
    zcr = float(np.sum(np.abs(np.diff(np.sign(mono)))) / (2.0 * len(mono))) if len(mono) > 0 else 0.0
    fb['zcr_history'].append(zcr)
    if len(fb['zcr_history']) > 10:
        fb['zcr_history'].pop(0)

    # Coupling field
    coupling.deposit(rms * 2.0)
    coupling.tick()
    extra = coupling.read()
    out *= (1.0 + extra * 0.3)

    # Phase 3: delay network
    out[:] = delay_net.process(out.T).T

    # Cold start
    fb['silence_counter'] += frames
    if rms > 1e-4:
        fb['silence_counter'] = 0
    if fb['silence_counter'] > int(SAMPLE_RATE * 0.5):
        noise = np.random.randn(frames).astype(np.float32) * 0.01
        out[:, 0] += noise
        out[:, 1] += noise
        fb['silence_counter'] = 0

    # Slow drift
    if 'sigma_drift' not in fb:
        fb['sigma_drift'] = 0.0
    fb['sigma_drift'] += np.random.randn() * 0.002
    fb['sigma_drift'] = np.clip(fb['sigma_drift'], -1.0, 1.0)
    chaos.sigma = np.clip(10.0 + fb['sigma_drift'], 6.0, 15.0)

    output[written:written+frames] = out
    written += frames
    if written % SAMPLE_RATE == 0:
        print(f"  {written//SAMPLE_RATE}s", flush=True)

# Normalize
peak = float(np.abs(output).max())
if peak > 0:
    output /= peak * 1.05
output_int16 = (np.clip(output, -1, 1) * 32767).astype(np.int16)
wavfile.write(OUTPUT, SAMPLE_RATE, output_int16)
print(f"Saved: {OUTPUT} ({os.path.getsize(OUTPUT)//1024}KB)", flush=True)
