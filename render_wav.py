#!/usr/bin/env python3
"""Offline render: Chaos Synth -> WAV file (Phase 2)"""
import sys, os
os.chdir('/Users/mengwu/Documents/Code/chaos-synth')
sys.path.insert(0, 'src')

import numpy as np
from scipy.io import wavfile
from main import *

DURATION = 10
OUTPUT = "/Users/mengwu/Documents/Code/chaos-synth/output/phase2_render.wav"
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

chaos = LorenzAttractor()
manifold = ManifoldMapper(n_centroids=16)
pool = VoicePool(capacity=128, max_active=16)
coupling = CouplingField()
ca = CellularAutomaton(width=64, rule=30)
lsys = LSystem("ABCB", {"A": "ABC", "B": "BAB", "C": "CA"})

trigger_gates = [True] * 8
ca_counter = ls_counter = 0
fb = {'centroid_history': [], 'flux_history': [], 'zcr_history': [], 'silence_counter': 0}

total_frames = DURATION * SAMPLE_RATE
output = np.zeros((total_frames, 2), dtype=np.float32)
written = 0

print(f"Rendering {DURATION}s @ {SAMPLE_RATE}Hz...", flush=True)

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
    freq = map_state_to_freq(state)
    if melody_notes and np.random.random() < 0.2:
        freq = melody_notes[np.random.randint(0, len(melody_notes))]
    amp = map_state_to_amp(state)
    if not trigger_gates[int(state[2]*8) % 8]:
        amp *= 0.3
    pool.trigger(eid, bid, mid, freq, amp, float(state[2]))

    buf = out.T.copy()
    pool.render(buf)
    out[:] = buf.T

    mono = out.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono**2)))
    coupling.deposit(rms * 2.0)
    coupling.tick()
    out *= (1.0 + coupling.read() * 0.3)

    fb['silence_counter'] += frames
    if rms > 1e-4: fb['silence_counter'] = 0
    if fb['silence_counter'] > int(SAMPLE_RATE * 0.5):
        noise = np.random.randn(frames).astype(np.float32) * 0.01
        out[:, 0] += noise; out[:, 1] += noise
        fb['silence_counter'] = 0

    if 'sigma_drift' not in fb: fb['sigma_drift'] = 0.0
    fb['sigma_drift'] += np.random.randn() * 0.002
    fb['sigma_drift'] = np.clip(fb['sigma_drift'], -1.0, 1.0)
    chaos.sigma = np.clip(10.0 + fb['sigma_drift'], 6.0, 15.0)

    output[written:written+frames] = out
    written += frames
    if written % SAMPLE_RATE == 0:
        print(f"  {written//SAMPLE_RATE}s", flush=True)

# Normalize
peak = float(np.abs(output).max())
if peak > 0: output /= peak * 1.05
output_int16 = (np.clip(output, -1, 1) * 32767).astype(np.int16)
wavfile.write(OUTPUT, SAMPLE_RATE, output_int16)
print(f"Saved: {OUTPUT} ({os.path.getsize(OUTPUT)//1024}KB)", flush=True)
