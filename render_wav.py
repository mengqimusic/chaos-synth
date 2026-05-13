#!/usr/bin/env python3
"""Offline render: Chaos Synth → WAV file"""
import sys, os
os.chdir('/Users/mengwu/Documents/Code/chaos-synth')
sys.path.insert(0, 'src')

import numpy as np
from scipy.io import wavfile
from main import *

# Config
DURATION = 15  # seconds
OUTPUT = "/Users/mengwu/Documents/Code/chaos-synth/output/phase2_render.wav"
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# Setup engine
chaos = LorenzAttractor()
manifold = ManifoldMapper(n_centroids=16)
pool = VoicePool(capacity=128, max_active=16)
coupling = CouplingField()
ca = CellularAutomaton(width=64, rule=30)
lsys = LSystem("ABCB", {"A": "ABC", "B": "BAB", "C": "CA"})

trigger_gates = [True] * 8
ca_counter, ls_counter = 0, 0
feedback_state = {
    'centroid_history': [],
    'flux_history': [],
    'zcr_history': [],
    'silence_counter': 0,
}

total_frames = DURATION * SAMPLE_RATE
output = np.zeros((total_frames, 2), dtype=np.float32)
written = 0

print(f"Rendering {DURATION}s @ {SAMPLE_RATE}Hz...")

while written < total_frames:
    frames = min(BLOCK_SIZE, total_frames - written)
    out = np.zeros((frames, 2), dtype=np.float32)

    # CA + LS step
    ca_counter += 1
    if ca_counter % 4 == 0:
        ca.step()
        trigger_gates = ca.trigger_pattern()
    ls_counter += 1
    if ls_counter % 20 == 0:
        lsys.iterate()
    melody_notes = lsys.to_notes(110.0) if ls_counter % 20 == 0 else []

    # Chaos step + trigger
    state = chaos.step()
    eid, bid, mid = manifold.find_nearest(state)
    gate_idx = int(state[2] * 8) % 8
    freq = map_state_to_freq(state)
    if melody_notes and np.random.rand() < 0.2:
        freq = melody_notes[np.random.randint(0, len(melody_notes))]
    amp = map_state_to_amp(state)
    if not trigger_gates[gate_idx]:
        amp *= 0.3
    pool.trigger(eid, bid, mid, freq, amp, float(state[2]))

    # Render
    buf_stereo = out.T.copy()
    pool.render(buf_stereo)
    out = buf_stereo.T

    # Feedback
    mono = out.mean(axis=1)
    rms = np.sqrt(np.mean(mono ** 2))
    coupling.deposit(rms * 2.0)
    coupling.tick()
    extra = coupling.read()
    out *= (1.0 + extra * 0.3)

    # Silence detection
    feedback_state['silence_counter'] += frames
    if rms > 1e-4:
        feedback_state['silence_counter'] = 0
    if feedback_state['silence_counter'] > int(SAMPLE_RATE * 0.5):
        noise = np.random.randn(frames).astype(np.float32) * 0.01
        out[:, 0] += noise
        out[:, 1] += noise
        feedback_state['silence_counter'] = 0

    # Lorenz sigma drift
    if 'sigma_drift' not in feedback_state:
        feedback_state['sigma_drift'] = 0.0
    feedback_state['sigma_drift'] += np.random.randn() * 0.002
    feedback_state['sigma_drift'] = np.clip(feedback_state['sigma_drift'], -1.0, 1.0)
    chaos.sigma = np.clip(10.0 + feedback_state['sigma_drift'], 6.0, 15.0)

    output[written:written+frames] = out
    written += frames

    if written % (SAMPLE_RATE * 5) == 0:
        print(f"  {written // SAMPLE_RATE}s / {DURATION}s")

# Normalize + save
peak = np.abs(output).max()
if peak > 0:
    output /= peak * 1.1  # -0.8dB headroom
output_int16 = (output * 32767).astype(np.int16)
wavfile.write(OUTPUT, SAMPLE_RATE, output_int16)

print(f"✓ Saved: {OUTPUT}")
print(f"  Duration: {DURATION}s, Peak: {peak:.3f}, Shape: {output_int16.shape}")
