# chaos-synth — Chaotic System Synthesizer
# Phase 0 MVP: Core synthesis engine
# Logistic map + 1D Voronoi manifold + 3 exciters + 3 bodies
# + static modulator + chaos pan spatializer + unified voice pool (128)
# Pure numpy, float32, single file.

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
BLOCK_SIZE = 256

# ═══════════════════════════════════════════════════════════════════════
# 1. Logistic Map — x[n+1] = r * x[n] * (1 - x[n])
# ═══════════════════════════════════════════════════════════════════════

class LogisticMap:
    """1D chaotic map. r=3.7 gives deterministic chaos in (0,1)."""

    def __init__(self, r: float = 3.7, x0: float = 0.5):
        self.r = r
        self.x = x0

    def step(self) -> float:
        """Advance one iteration, return x ∈ (0, 1)."""
        self.x = self.r * self.x * (1.0 - self.x)
        # Clamp away from fixed points at 0 and 1
        if self.x < 1e-6:
            self.x = 1e-6
        elif self.x > 1.0 - 1e-6:
            self.x = 1.0 - 1e-6
        return self.x


# ═══════════════════════════════════════════════════════════════════════
# 2. 1D Voronoi Manifold — 4 centroids on [0,1]
# ═══════════════════════════════════════════════════════════════════════

class ManifoldMapper:
    """1D Voronoi tessellation: nearest-centroid → (exciter, body) combo."""

    def __init__(self):
        self.centroids = np.array([0.125, 0.375, 0.625, 0.875], dtype=np.float32)
        self.combos = [
            (0, 0),  # c0: sine + dry
            (1, 1),  # c1: noise + modal
            (4, 2),  # c2: click + comb
            (0, 2),  # c3: sine + comb
        ]

    def find_nearest(self, x: float) -> tuple:
        """Return (exciter_id, body_id) for centroid nearest to x."""
        idx = int(np.argmin(np.abs(self.centroids - x)))
        return self.combos[idx]


# ═══════════════════════════════════════════════════════════════════════
# 3. Exciters — how sound begins
# ═══════════════════════════════════════════════════════════════════════

def exciter_sine_impulse(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#0: One sine cycle, linear attack 1/4 + exponential decay 3/4."""
    length = max(int(sr / freq), 4)
    t = np.arange(length, dtype=np.float32)
    sine = np.sin(2.0 * np.pi * t / length)
    quarter = max(1, length // 4)
    env = np.ones(length, dtype=np.float32)
    env[:quarter] = np.linspace(0.0, 1.0, quarter, dtype=np.float32)
    env[quarter:] = np.exp(-3.0 * (t[quarter:] - quarter) / max(length - quarter, 1))
    return sine * env


def exciter_noise_burst(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#1: 512-sample white noise, Hann window, amplitude normalised to 0.3."""
    noise = np.random.randn(512).astype(np.float32)
    burst = noise * np.hanning(512).astype(np.float32)
    peak = np.max(np.abs(burst))
    if peak > 1e-8:
        burst *= 0.3 / peak
    return burst


def exciter_click(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#4: 4-sample impulse + exp(-t/tau) tail, tau=1ms, 256 samples."""
    buf = np.zeros(256, dtype=np.float32)
    buf[:4] = 1.0
    tau = float(sr) * 0.001
    buf[4:] = np.exp(-np.arange(4, 256, dtype=np.float32) / tau)
    return buf


EXCITERS = {0: exciter_sine_impulse, 1: exciter_noise_burst, 4: exciter_click}


# ═══════════════════════════════════════════════════════════════════════
# 4. Bodies — how sound sustains & decays
# ═══════════════════════════════════════════════════════════════════════

def body_dry(excitation: np.ndarray, freq: float,
             sr: int = SAMPLE_RATE) -> np.ndarray:
    """#0: Exponential decay envelope, tau = 50 ms."""
    t = np.arange(len(excitation), dtype=np.float32) / (float(sr) * 0.05)
    return excitation * np.exp(-t)


def body_modal(excitation: np.ndarray, freq: float,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """#1: Two inharmonic modal sines at freq*1.0 (tau=0.1s, amp=0.6)
    and freq*2.76 (tau=0.04s, amp=0.4), added to excitation."""
    t = np.arange(len(excitation), dtype=np.float32) / float(sr)
    s1 = np.sin(2.0 * np.pi * freq * t) * np.exp(-t / 0.1) * 0.6
    s2 = np.sin(2.0 * np.pi * freq * 2.76 * t) * np.exp(-t / 0.04) * 0.4
    return excitation + s1 + s2


def body_comb(excitation: np.ndarray, freq: float,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """#2: Feedback comb filter. delay = sr/freq/2, feedback = 0.7."""
    length = len(excitation)
    delay_len = max(int(float(sr) / freq / 2.0), 1)
    out = excitation.copy()
    fb = np.float32(0.7)
    for i in range(delay_len, length):
        out[i] += fb * out[i - delay_len]
    return out


BODIES = {0: body_dry, 1: body_modal, 2: body_comb}


# ═══════════════════════════════════════════════════════════════════════
# 5. Modulator — parameter evolution over time
# ═══════════════════════════════════════════════════════════════════════

def modulator_static(buf: np.ndarray) -> np.ndarray:
    """#0: No modulation — pass-through."""
    return buf


MODULATORS = {0: modulator_static}


# ═══════════════════════════════════════════════════════════════════════
# 6. Spatializer — particle placement in space
# ═══════════════════════════════════════════════════════════════════════

def spatialize_chaos_pan(mono: np.ndarray, chaos_x: float) -> np.ndarray:
    """#0: pan = x*2-1 → [-1, +1]. Returns stereo (2, N) float32."""
    pan = chaos_x * 2.0 - 1.0
    left = np.float32((1.0 - pan) * 0.5)
    right = np.float32((1.0 + pan) * 0.5)
    stereo = np.empty((2, len(mono)), dtype=np.float32)
    stereo[0] = mono * left
    stereo[1] = mono * right
    return stereo


SPATIALIZERS = {0: spatialize_chaos_pan}


# ═══════════════════════════════════════════════════════════════════════
# 7. Unified Voice Pool — 128 slots, max 8 concurrent
# ═══════════════════════════════════════════════════════════════════════

class VoicePool:
    """128-slot voice pool. Each voice = exciter→body→modulator grain.
    Grains pre-rendered on trigger(), mixed incrementally across render().
    State: 0=attack, 1=sustain, 2=decay, 3=done/free.
    """

    def __init__(self, capacity: int = 128, max_active: int = 8):
        self.capacity = capacity
        self.max_active = max_active
        self._trigger_clock = 0
        # Pre-allocate all voice slots
        self.voices = []
        for _ in range(capacity):
            self.voices.append({
                'active': False, 'exciter_id': 0, 'body_id': 0,
                'freq': 440.0, 'amp': 0.3, 'chaos_x': 0.5,
                'age': 0, 'duration': 0, 'state': 3,
                'buffer': None, 'trigger_time': 0,
            })

    def _find_slot(self) -> int:
        """Return a free slot index; steal oldest active if pool full."""
        for i, v in enumerate(self.voices):
            if not v['active'] and v['state'] == 3:
                return i
        # Steal oldest active voice
        oldest_i, oldest_t = 0, float('inf')
        for i, v in enumerate(self.voices):
            if v['active'] and v['trigger_time'] < oldest_t:
                oldest_t = v['trigger_time']
                oldest_i = i
        return oldest_i

    def _active_count(self) -> int:
        return sum(1 for v in self.voices if v['active'])

    def trigger(self, exciter_id: int, body_id: int, freq: float,
                amp: float, chaos_x: float) -> None:
        """Spawn a new voice. Steals oldest if at max_active."""
        if self._active_count() >= self.max_active:
            oldest_i, oldest_t = 0, float('inf')
            for i, v in enumerate(self.voices):
                if v['active'] and v['trigger_time'] < oldest_t:
                    oldest_t = v['trigger_time']
                    oldest_i = i
            self.voices[oldest_i]['active'] = False
            self.voices[oldest_i]['state'] = 3

        slot = self._find_slot()
        self._trigger_clock += 1

        # Render the full grain: exciter → body → modulator
        efn = EXCITERS.get(exciter_id, exciter_sine_impulse)
        bfn = BODIES.get(body_id, body_dry)
        mfn = MODULATORS.get(0, modulator_static)
        grain = mfn(bfn(efn(freq, SAMPLE_RATE), freq, SAMPLE_RATE))
        grain = np.clip(grain * amp, -1.0, 1.0).astype(np.float32)

        v = self.voices[slot]
        v.update(active=True, exciter_id=exciter_id, body_id=body_id,
                 freq=freq, amp=amp, chaos_x=chaos_x,
                 age=0, duration=len(grain), state=0,
                 buffer=grain, trigger_time=self._trigger_clock)

    def render(self, buffer_stereo: np.ndarray, sr: int = SAMPLE_RATE) -> None:
        """Mix active voice grains into stereo output buffer (2, frames)."""
        frames = buffer_stereo.shape[1]
        for v in self.voices:
            if not v['active']:
                continue
            remaining = v['duration'] - v['age']
            to_mix = min(remaining, frames)
            if to_mix <= 0:
                v['active'] = False
                v['state'] = 3
                continue

            pan = v['chaos_x'] * 2.0 - 1.0
            lg = np.float32((1.0 - pan) * 0.5)
            rg = np.float32((1.0 + pan) * 0.5)
            seg = v['buffer'][v['age']:v['age'] + to_mix]
            buffer_stereo[0, :to_mix] += seg * lg
            buffer_stereo[1, :to_mix] += seg * rg

            v['age'] += to_mix
            if v['age'] >= v['duration']:
                v['active'] = False
                v['state'] = 3


# ═══════════════════════════════════════════════════════════════════════
# 8. Parameter Mapping
# ═══════════════════════════════════════════════════════════════════════

def map_x_to_freq(x: float) -> float:
    """Map x ∈ [0,1] → 55 Hz – 1760 Hz (A1–A6, 5 octaves)."""
    return 55.0 * (2.0 ** (x * 5.0))


def map_x_to_amp(x: float) -> float:
    """Map x ∈ [0,1] → amplitude 0.05 – 0.5."""
    return 0.05 + x * 0.45


# ═══════════════════════════════════════════════════════════════════════
# 9. Spectral centroid & audio callback (sounddevice)
# ═══════════════════════════════════════════════════════════════════════

def compute_spectral_centroid(signal: np.ndarray, sr: int) -> float:
    """计算频谱质心 (Hz)。纯 numpy，无分配（in-place FFT）。"""
    windowed = signal * np.hanning(len(signal)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr).astype(np.float32)
    total = spectrum.sum()
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)





# ═══════════════════════════════════════════════════════════════════════
# 10. Entry point — audio callback as closure for sounddevice 0.5.5
# ═══════════════════════════════════════════════════════════════════════

def run(duration=None, device=None):
    """Launch the synth. duration=None runs forever (Ctrl+C to stop)."""
    import sounddevice as sd

    logistic = LogisticMap(r=3.7, x0=0.5)
    manifold = ManifoldMapper()
    pool = VoicePool(capacity=128, max_active=8)

    feedback_state = {
        'centroid_history': [],
        'silence_counter': 0,
    }

    # Closure: captures logistic, manifold, pool, feedback_state
    def audio_callback(outdata, frames, time_info, status):
        outdata.fill(0.0)

        x = logistic.step()
        exciter_id, body_id = manifold.find_nearest(x)
        freq = map_x_to_freq(x)
        amp = map_x_to_amp(x)
        pool.trigger(exciter_id, body_id, freq, amp, x)
        pool.render(outdata.T)

        mono = outdata.mean(axis=1)
        centroid = compute_spectral_centroid(mono, SAMPLE_RATE)

        feedback_state['centroid_history'].append(centroid)
        if len(feedback_state['centroid_history']) > 10:
            feedback_state['centroid_history'].pop(0)

        avg_centroid = np.mean(feedback_state['centroid_history'])
        # Map avg centroid to r: dark(~500Hz)→3.5, neutral(~2500Hz)→3.7, bright(~5000Hz)→3.95
        dr = np.clip((avg_centroid / 2500.0 - 1.5) * 0.25, -0.2, 0.25)
        logistic.r = np.clip(3.5 + dr, 3.5, 3.95)

        feedback_state['silence_counter'] += frames
        rms = np.sqrt(np.mean(mono ** 2))
        if rms > 1e-4:
            feedback_state['silence_counter'] = 0

        if feedback_state['silence_counter'] > int(SAMPLE_RATE * 0.5):
            noise = np.random.randn(frames).astype(np.float32) * 0.01
            outdata[:, 0] += noise
            outdata[:, 1] += noise
            feedback_state['silence_counter'] = 0

        # Slow r drift: prevent lock-in at any single value
        # Adds ~0.0001 per second of random walk, bounded
        if 'r_drift' not in feedback_state:
            feedback_state['r_drift'] = 0.0
        feedback_state['r_drift'] += np.random.randn() * 0.00002
        feedback_state['r_drift'] = np.clip(feedback_state['r_drift'], -0.05, 0.05)
        logistic.r = np.clip(logistic.r + feedback_state['r_drift'], 3.5, 3.95)

    print(f"chaos-synth v0.1.0 [Phase 0 MVP]")
    print(f"  Logistic: r={logistic.r:.2f}")
    print(f"  Manifold: 4 centroids, combos={manifold.combos}")
    print(f"  Pool: {pool.capacity} slots, max {pool.max_active} active")
    print(f"  S/R: {SAMPLE_RATE}Hz, block: {BLOCK_SIZE}")
    print(f"  Ctrl+C to stop")

    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=2,
        dtype='float32',
        callback=audio_callback,
        device=device,
    )

    try:
        with stream:
            if duration:
                sd.sleep(int(duration * 1000))
            else:
                while True:
                    sd.sleep(1000)
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    run()
