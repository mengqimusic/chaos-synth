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
# ═══════════════════════════════════════════════════════════════════════
# 1. Chaos Engine - Logistic / Lorenz / Roessler (Phase 1)
# ═══════════════════════════════════════════════════════════════════════

class LogisticMap:
    """1D chaotic map. r=3.7 gives deterministic chaos. Returns array([x])."""

    def __init__(self, r: float = 3.7, x0: float = 0.5):
        self.r = r
        self._x = x0

    def step(self) -> np.ndarray:
        self._x = self.r * self._x * (1.0 - self._x)
        if self._x < 1e-6:
            self._x = 1e-6
        elif self._x > 1.0 - 1e-6:
            self._x = 1.0 - 1e-6
        return np.array([self._x], dtype=np.float32)


class LorenzAttractor:
    """3D Lorenz attractor. Returns array([x, y, z]) normalized to [0,1]."""

    def __init__(self, sigma=10.0, rho=28.0, beta=2.667, dt=0.01):
        self.sigma, self.rho, self.beta, self.dt = sigma, rho, beta, dt
        self.x, self.y, self.z = 1.0, 1.0, 1.0

    def step(self) -> np.ndarray:
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        return np.array([
            np.clip((self.x + 20.0) / 40.0, 0, 1),
            np.clip((self.y + 25.0) / 50.0, 0, 1),
            np.clip(self.z / 50.0, 0, 1),
        ], dtype=np.float32)


class RoesslerAttractor:
    """3D Roessler attractor. Returns array([x, y, z]) normalized to [0,1]."""

    def __init__(self, a=0.2, b=0.2, c=5.7, dt=0.03):
        self.a, self.b, self.c, self.dt = a, b, c, dt
        self.x, self.y, self.z = 1.0, 1.0, 1.0

    def step(self) -> np.ndarray:
        dx = -self.y - self.z
        dy = self.x + self.a * self.y
        dz = self.b + self.z * (self.x - self.c)
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.z += dz * self.dt
        return np.array([
            np.clip((self.x + 10.0) / 25.0, 0, 1),
            np.clip((self.y + 10.0) / 25.0, 0, 1),
            np.clip(self.z / 25.0, 0, 1),
        ], dtype=np.float32)


CHAOS_ENGINES = {
    'logistic': LogisticMap,
    'lorenz': LorenzAttractor,
    'roessler': RoesslerAttractor,
}
# ═══════════════════════════════════════════════════════════════════════
# 2. 3D Voronoi Manifold — 16 centroids on [0,1]³
# ═══════════════════════════════════════════════════════════════════════

class ManifoldMapper:
    """3D Voronoi tessellation: nearest-centroid → (exciter, body, mod)."""

    def __init__(self, n_centroids: int = 32, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.centroids = rng.rand(n_centroids, 3).astype(np.float32)
        # Assign combos spread across full module space
        self.combos = []
        for i in range(n_centroids):
            e = i % 12  # exciter 0-11
            b = i % 10  # body 0-9
            m = i % 7   # modulator 0-6
            self.combos.append((e, b, m))

    def find_nearest(self, point: np.ndarray) -> tuple:
        """Return (exciter_id, body_id, modulator_id)."""
        dists = np.sum((self.centroids - point.astype(np.float32)) ** 2, axis=1)
        idx = int(np.argmin(dists))
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


def exciter_fm_spark(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#2: FM spark. carrier=freq, modulator=freq*2, mod_idx=3. 256 samples."""
    t = np.arange(256, dtype=np.float32) / float(sr)
    mod = np.sin(2.0 * np.pi * freq * 2.0 * t) * 3.0
    sig = np.sin(2.0 * np.pi * freq * t + mod)
    env = np.exp(-t * 40.0)
    return (sig * env).astype(np.float32)


def exciter_granular_micro(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#3: Granular micro. 4 sub-grains, clipped to 128 max per grain."""
    out = np.zeros(256, dtype=np.float32)
    for i in range(4):
        g_len = min(np.random.randint(max(4, int(sr * 0.003)), int(sr * 0.008)), 128)
        g_start = np.random.randint(0, max(1, 256 - g_len))
        g_pitch = freq * (0.8 + np.random.rand() * 0.4)
        t = np.arange(g_len, dtype=np.float32) / float(sr)
        g = np.sin(2.0 * np.pi * g_pitch * t) * np.hanning(g_len).astype(np.float32)
        out[g_start:g_start + g_len] += g[:g_len] * 0.25
    return out


def exciter_chirp(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#5: Frequency sweep from freq to freq*4, 256 samples."""
    t = np.arange(256, dtype=np.float32) / float(sr)
    phase = 2.0 * np.pi * freq * (t + 3.0 * t ** 2 * freq)
    sig = np.sin(phase)
    env = np.exp(-t * 15.0)
    return (sig * env).astype(np.float32)


def exciter_pluck(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#6: Physical pluck (Karplus-Strong). Dynamic buffer, min 512 samples."""
    delay = max(int(sr / freq), 4)
    buf_size = max(delay + 4, 512)  # ensure buffer fits the initial noise fill
    out = np.zeros(buf_size, dtype=np.float32)
    out[:delay] = (np.random.rand(delay).astype(np.float32) * 2.0 - 1.0) * 0.5
    for i in range(delay, buf_size):
        out[i] = (out[i - delay] + out[i - delay + 1]) * 0.5 * 0.99
    out *= np.exp(-np.arange(buf_size, dtype=np.float32) / (float(sr) * 0.03))
    return out


def exciter_wavetable(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#7: Wavetable scan. Sine sweep through variable shapes, 512 samples."""
    length = 512
    t = np.arange(length, dtype=np.float32)
    phase = t * freq / float(sr)
    # Scan from sine to saw by mixing
    mix = t / float(length)  # 0->1 over the grain
    sine = np.sin(2.0 * np.pi * phase)
    saw = 2.0 * (phase - np.floor(phase + 0.5))
    return ((1.0 - mix) * sine + mix * saw).astype(np.float32)



def exciter_fb_ping(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#8: Feedback ping — short self-oscillation at critical feedback."""
    length = max(int(sr / freq), 8)
    buf = np.zeros(length, dtype=np.float32)
    buf[0] = 1.0
    fb = 0.92 + np.random.randn() * 0.03  # near self-oscillation
    for i in range(1, length):
        buf[i] = buf[i-1] * fb
        if i > 1: buf[i] = buf[i] * 0.99 + buf[i-2] * 0.01
    return buf * np.exp(-np.arange(length, dtype=np.float32) / (float(sr) * 0.01))


def exciter_vowel(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#9: Vowel burst — formant-filtered noise, 256 samples."""
    noise = np.random.randn(256).astype(np.float32) * 0.3
    # Simple 2-formant resonator (F1~500, F2~1500 Hz)
    t = np.arange(256, dtype=np.float32) / float(sr)
    f1, f2 = 500.0 + np.random.randn() * 100, 1500.0 + np.random.randn() * 200
    res = np.sin(2*np.pi*f1*t) * 0.6 + np.sin(2*np.pi*f2*t) * 0.4
    return (noise + res * 0.3) * np.hanning(256).astype(np.float32) * 0.5


def exciter_ringmod(freq: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """#10: Ringmod spike — two very high freqs multiplied, 3-10ms."""
    length = min(max(int(sr * 0.005), 32), 256)
    t = np.arange(length, dtype=np.float32) / float(sr)
    f1 = freq * (2.0 + np.random.rand() * 6.0)
    f2 = f1 * 1.07  # slight detune
    sig = np.sin(2*np.pi*f1*t) * np.sin(2*np.pi*f2*t)
    return sig * np.exp(-t * 200.0)


def exciter_transient(freq: float, sr: int = SAMPLE_RATE,
                      self_sample: np.ndarray = None) -> np.ndarray:
    """#11: Transient snatch from self-sampling ring buffer (Phase 3)."""
    if self_sample is not None and len(self_sample) > 0:
        # Use real self-sampled audio as the grain
        return self_sample.astype(np.float32) * 0.8
    # Fallback: short burst of filtered noise
    buf = np.random.randn(128).astype(np.float32) * 0.2 * np.hanning(128).astype(np.float32)
    return buf


EXCITERS = {
    0: exciter_sine_impulse, 1: exciter_noise_burst, 2: exciter_fm_spark,
    3: exciter_granular_micro, 4: exciter_click, 5: exciter_chirp,
    6: exciter_pluck, 7: exciter_wavetable,
    8: exciter_fb_ping, 9: exciter_vowel, 10: exciter_ringmod, 11: exciter_transient,
}


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


def body_allpass(excitation: np.ndarray, freq: float,
                 sr: int = SAMPLE_RATE) -> np.ndarray:
    """#3: Cascaded allpass filter (reverb tail feel). 4 stages."""
    delay = max(int(float(sr) / freq / 4.0), 2)
    g = 0.5
    out = excitation.copy().astype(np.float32)
    for _ in range(4):
        buf = np.zeros(delay, dtype=np.float32)
        for i in range(len(out)):
            dly = buf[i % delay]
            buf[i % delay] = out[i] + g * dly
            out[i] = -g * out[i] + dly
    return out


def body_nonlinear(excitation: np.ndarray, freq: float,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """#4: Saturation chain. 3 stages of tanh distortion."""
    out = excitation.copy().astype(np.float32)
    for _ in range(3):
        out = np.tanh(out * 2.0) * 1.2
    return out


def body_freeze(excitation: np.ndarray, freq: float,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """#5: Granular freeze. Repeat first 64 samples, 3 iterations."""
    loop = excitation[:min(64, len(excitation))].copy()
    out = excitation.copy()
    for i in range(3):
        start = min(len(out), len(loop))
        end = min(len(out), start + 64)
        out[start:end] += loop[:end - start] * 0.3
    return out



def body_waveguide(excitation: np.ndarray, freq: float,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """#6: 1D Waveguide — tube/string resonator. Delay = sr/(2*freq)."""
    length = len(excitation)
    delay = max(int(sr / freq / 2.0), 2)
    out = excitation.copy().astype(np.float32)
    for i in range(delay, length):
        out[i] += (out[i-delay] - out[i-delay+1] if i-delay+1 < length else 0) * 0.5 * 0.97
    return out


def body_saturation(excitation: np.ndarray, freq: float,
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    """#7: 3-stage tanh saturation chain with increasing drive."""
    out = excitation.copy().astype(np.float32)
    for n in range(3):
        out = np.tanh(out * (1.5 + n * 1.5)) * 0.8
    return out


def body_blur(excitation: np.ndarray, freq: float,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """#8: Spectral blur — phase randomization in FFT domain."""
    fx = np.fft.rfft(excitation).astype(np.complex64)
    mag = np.abs(fx)
    random_phase = np.exp(1j * np.random.rand(len(mag)) * 2 * np.pi).astype(np.complex64)
    return np.fft.irfft(mag * random_phase, n=len(excitation)).astype(np.float32)


def body_pshift_fb(excitation: np.ndarray, freq: float,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """#9: Pitch-shifted feedback — shimmer effect (rising pitch trails)."""
    length = len(excitation)
    delay = max(int(sr / freq / 3.0), 2)
    out = excitation.copy().astype(np.float32)
    shift = 1.002  # slight pitch shift per iteration
    for i in range(delay, length, 2):
        idx = int((i - delay) * shift) % length
        out[i] += out[idx] * 0.5
    return out


BODIES = {0: body_dry, 1: body_modal, 2: body_comb, 3: body_allpass,
          4: body_nonlinear, 5: body_freeze,
    6: body_waveguide, 7: body_saturation, 8: body_blur, 9: body_pshift_fb,
}


# ═══════════════════════════════════════════════════════════════════════
# 5. Modulator — parameter evolution over time
# ═══════════════════════════════════════════════════════════════════════

def modulator_static(buf: np.ndarray) -> np.ndarray:
    """#0: No modulation — pass-through."""
    return buf


def modulator_tremolo(buf: np.ndarray, freq: float = 5.0) -> np.ndarray:
    """#1: Amplitude modulation at 5 Hz."""
    t = np.arange(len(buf), dtype=np.float32) / SAMPLE_RATE
    lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * freq * t)
    return (buf * lfo).astype(np.float32)


def modulator_vibrato(buf: np.ndarray, freq: float = 5.0, depth: float = 0.003) -> np.ndarray:
    """#2: Frequency modulation (done as phase offset on delay). 256-sample buffer.
    Simple implementation: modulate linear interpolation within buffer."""
    n = len(buf)
    t = np.arange(n, dtype=np.float32)
    mod = 1.0 + depth * np.sin(2.0 * np.pi * freq * t / SAMPLE_RATE * n)
    indices = np.clip(np.arange(n, dtype=np.float32) * mod, 0, n - 1).astype(np.int32)
    return (buf[indices]).astype(np.float32)


def modulator_phase_dist(buf: np.ndarray, distortion: float = 0.5) -> np.ndarray:
    """#3: Phase distortion. Read buffer through a bent phase (Casio CZ style)."""
    n = len(buf)
    phase = np.arange(n, dtype=np.float32) / float(n)
    bent = phase + distortion * np.sin(2.0 * np.pi * phase)
    bent = bent / bent.max() * (n - 1)
    idx = bent.astype(np.int32)
    frac = bent - idx
    idx = np.clip(idx, 0, n - 2)
    return (buf[idx] * (1.0 - frac) + buf[idx + 1] * frac).astype(np.float32)



def modulator_ringmod(buf: np.ndarray, freq: float = 200.0) -> np.ndarray:
    """#4: Ring modulation with fixed carrier."""
    t = np.arange(len(buf), dtype=np.float32) / SAMPLE_RATE
    carrier = np.sin(2.0 * np.pi * freq * t)
    return (buf * carrier).astype(np.float32)


def modulator_bitcrush(buf: np.ndarray, bits: int = 6) -> np.ndarray:
    """#5: Bit reduction for lo-fi texture."""
    levels = 2.0 ** bits
    return (np.round(buf * levels) / levels).astype(np.float32)


def modulator_stereo_width(buf: np.ndarray, width: float = 0.8) -> np.ndarray:
    """#6: Stereo widening via Mid/Side processing. Returns (2, N) stereo."""
    mid = buf
    side = buf * width
    stereo = np.empty((2, len(buf)), dtype=np.float32)
    stereo[0] = (mid + side) * 0.5
    stereo[1] = (mid - side) * 0.5
    return stereo


MODULATORS = {0: modulator_static, 1: modulator_tremolo,
              2: modulator_vibrato, 3: modulator_phase_dist,
    4: modulator_ringmod, 5: modulator_bitcrush, 6: modulator_stereo_width,
}


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



def spatialize_haas(mono: np.ndarray, chaos_x: float) -> np.ndarray:
    """#1: Haas effect — small delay offset (0-30ms) for spatialization."""
    delay_samples = int(abs(chaos_x) * SAMPLE_RATE * 0.03)
    delay_samples = min(delay_samples, len(mono) - 1)
    stereo = np.zeros((2, len(mono)), dtype=np.float32)
    if chaos_x > 0:
        stereo[0, :len(mono)-delay_samples] = mono[delay_samples:]
        stereo[1] = mono
    else:
        stereo[0] = mono
        stereo[1, delay_samples:] = mono[:len(mono)-delay_samples]
    return stereo


def spatialize_distance(mono: np.ndarray, chaos_x: float) -> np.ndarray:
    """#2: Distance decay — lowpass + amplitude reduction for depth."""
    dist = 0.3 + abs(chaos_x) * 0.7  # 0.3-1.0
    # Simple lowpass via moving average
    window = max(1, int(SAMPLE_RATE / 4000.0))
    kernel = np.ones(window, dtype=np.float32) / window
    filtered = np.convolve(mono, kernel, mode='same').astype(np.float32)
    pan = chaos_x * 2.0 - 1.0
    l, r = (1.0-pan)*0.5, (1.0+pan)*0.5
    stereo = np.empty((2, len(mono)), dtype=np.float32)
    stereo[0] = filtered * l * dist
    stereo[1] = filtered * r * dist
    return stereo


SPATIALIZERS = {0: spatialize_chaos_pan, 1: spatialize_haas, 2: spatialize_distance}



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

    def trigger(self, exciter_id: int, body_id: int, modulator_id: int,
                freq: float, amp: float, chaos_x: float,
                self_sample: np.ndarray = None) -> None:
        """Spawn a new voice. Steals oldest if at max_active.
        self_sample: optional buffer for exciter #11 transient snatch."""
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
        if exciter_id == 11 and self_sample is not None and len(self_sample) > 0:
            # Use self-sampled audio for transient snatch
            excitation = self_sample.astype(np.float32) * 0.8
        else:
            efn = EXCITERS.get(exciter_id, exciter_sine_impulse)
            excitation = efn(freq, SAMPLE_RATE)
        bfn = BODIES.get(body_id, body_dry)
        grain = bfn(excitation, freq, SAMPLE_RATE)
        # Apply modulator
        if modulator_id == 1:
            grain = modulator_tremolo(grain, freq=5.0)
        elif modulator_id == 2:
            grain = modulator_vibrato(grain, freq=5.0, depth=0.003)
        elif modulator_id == 3:
            grain = modulator_phase_dist(grain, distortion=0.5)
        elif modulator_id == 4:
            grain = modulator_ringmod(grain, freq=200.0)
        elif modulator_id == 5:
            grain = modulator_bitcrush(grain, bits=6)
        elif modulator_id == 6:
            grain_buf = modulator_stereo_width(grain, width=0.8)
            # Stereo width outputs (2,N) — use it directly
            grain = grain_buf.mean(axis=0).astype(np.float32)
        else:
            grain = modulator_static(grain)
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
# ═══════════════════════════════════════════════════════════════════════
# 8. Parameter Mapping (3D state aware)
# ═══════════════════════════════════════════════════════════════════════

def map_state_to_freq(state: np.ndarray) -> float:
    """Map state[0] in [0,1] to 55 Hz - 1760 Hz (A1-A6, 5 octaves)."""
    return 55.0 * (2.0 ** (float(state[0]) * 5.0))


def map_state_to_amp(state: np.ndarray) -> float:
    """Map state[1] in [0,1] to amplitude 0.05 - 0.5."""
    return 0.05 + float(state[1]) * 0.45

# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# 9. Coupling Field — shared energy buffer (Phase 1)
# ═══════════════════════════════════════════════════════════════════════

class CouplingField:
    """~1s ring buffer. Voices deposit RMS energy, read back for coupling."""

    def __init__(self, length: int = 44100):
        self.field = np.zeros(length, dtype=np.float32)
        self.pos = 0

    def read(self, offset: int = 0) -> float:
        """Read field value at current position + offset."""
        return float(self.field[(self.pos + offset) % len(self.field)])

    def deposit(self, energy: float) -> None:
        """Add energy to current position, clamp to 1.0. Advance."""
        self.field[self.pos] = min(self.field[self.pos] + energy, 1.0)
        self.pos = (self.pos + 1) % len(self.field)

    def tick(self) -> None:
        """Apply slow decay."""
        self.field *= 0.9995


# ═══════════════════════════════════════════════════════════════════════
# 9b. Self-Sampling Ring Buffer (Phase 3)
# ═══════════════════════════════════════════════════════════════════════

class SelfSampleBuffer:
    """2-second ring buffer recording stereo output. Exciter #11 reads from it."""

    def __init__(self, duration_s: float = 2.0, sr: int = SAMPLE_RATE):
        self.buffer = np.zeros((2, int(sr * duration_s)), dtype=np.float32)
        self.pos = 0
        self.length = self.buffer.shape[1]

    def write(self, stereo: np.ndarray) -> None:
        """Write a block of stereo audio into ring buffer."""
        n = stereo.shape[1]
        end = self.pos + n
        if end <= self.length:
            self.buffer[:, self.pos:end] = stereo
        else:
            split = self.length - self.pos
            self.buffer[:, self.pos:] = stereo[:, :split]
            self.buffer[:, :end - self.length] = stereo[:, split:]
        self.pos = end % self.length

    def snatch(self, length: int = 256) -> np.ndarray:
        """Grab a recent transient from buffer. Returns mono float32."""
        start = (self.pos - length) % self.length
        if start + length <= self.length:
            seg = self.buffer.mean(axis=0)[start:start + length].copy()
        else:
            seg = np.concatenate([
                self.buffer.mean(axis=0)[start:],
                self.buffer.mean(axis=0)[:length - (self.length - start)]
            ])
        return seg * np.hanning(length).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# 9c. Delay Network — 4-line cross-feedback (Phase 3)
# ═══════════════════════════════════════════════════════════════════════

class DelayNetwork:
    """4 parallel delay lines with cross-feedback matrix (no-input mixer style)."""

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        # Delay times: 150ms, 225ms, 337ms, 506ms (prime-related ratios)
        self.delays = [int(sr * t) for t in [0.150, 0.225, 0.337, 0.506]]
        self.buffers = [np.zeros(d, dtype=np.float32) for d in self.delays]
        self.positions = [0, 0, 0, 0]
        # Cross-feedback matrix (4x4) — each line feeds into others
        self.fb_matrix = np.array([
            [0.0,  0.3,  0.15, 0.1],
            [0.2,  0.0,  0.25, 0.15],
            [0.1,  0.2,  0.0,  0.3],
            [0.15, 0.1,  0.3,  0.0],
        ], dtype=np.float32)
        self.wet_mix = 0.3  # global wet/dry

    def set_feedback(self, amount: float):
        """Scale feedback matrix by amount (0-1)."""
        self.fb_matrix *= amount / (np.abs(self.fb_matrix).max() + 1e-8)

    def process(self, stereo_in: np.ndarray) -> np.ndarray:
        """Process one block through delay network. Returns stereo."""
        frames = stereo_in.shape[1]
        mono = stereo_in.mean(axis=0)
        out = np.zeros(frames, dtype=np.float32)

        for f in range(frames):
            sample_in = mono[f]
            sample_out = 0.0

            for i in range(4):
                # Read from delay line
                delayed = self.buffers[i][self.positions[i]]
                # Write input + cross-feedback
                fb_sum = 0.0
                for j in range(4):
                    if i != j:
                        fb_sum += self.fb_matrix[i, j] * self.buffers[j][self.positions[j]]
                self.buffers[i][self.positions[i]] = sample_in * 0.5 + fb_sum
                # Advance position
                self.positions[i] = (self.positions[i] + 1) % self.delays[i]
                # Mix output
                sample_out += delayed * 0.25

            out[f] = sample_out * self.wet_mix

        # Mix with original stereo
        stereo_out = stereo_in.copy()
        stereo_out[0] += out * 0.5
        stereo_out[1] += out * 0.5
        return np.clip(stereo_out, -1.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# 9d. Long-Term Feedback — 1s spectral averaging (Phase 3)
# ═══════════════════════════════════════════════════════════════════════

class LongTermFeedback:
    """Accumulates 1-second spectral averages, slowly modulates attractor params."""

    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.centroid_buffer = []
        self.flux_buffer = []
        self.window_size = sr // BLOCK_SIZE  # ~172 frames for 1 second
        # Current long-term targets (smoothed)
        self.lt_centroid = 500.0
        self.lt_flux = 50.0
        self.lt_rms = 0.05

    def feed(self, centroid: float, flux: float, rms: float):
        """Add one frame of features."""
        self.centroid_buffer.append(centroid)
        self.flux_buffer.append(flux)
        if len(self.centroid_buffer) > self.window_size:
            self.centroid_buffer.pop(0)
            self.flux_buffer.pop(0)

    def tick(self) -> dict:
        """Compute long-term averages and return modulation suggestions."""
        if len(self.centroid_buffer) < 10:
            return {}
        avg_c = np.mean(self.centroid_buffer)
        avg_f = np.mean(self.flux_buffer)
        # Smooth targets
        self.lt_centroid += (avg_c - self.lt_centroid) * 0.05
        self.lt_flux += (avg_f - self.lt_flux) * 0.05
        return {
            'lt_brightness': min(self.lt_centroid / 5000.0, 1.0),  # 0=dark, 1=bright
            'lt_activity': min(self.lt_flux / 500.0, 1.0),  # 0=calm, 1=busy
        }


# ═══════════════════════════════════════════════════════════════════════
# 9e. Spectral analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_spectral_centroid(signal: np.ndarray, sr: int) -> float:
    """Spectral centroid in Hz. Pure numpy, no allocs."""
    windowed = signal * np.hanning(len(signal)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr).astype(np.float32)
    total = spectrum.sum()
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


# ═══════════════════════════════════════════════════════════════════════
# 8b. Scale Definitions + Performance Handle Mapping (Phase 3)
# ═══════════════════════════════════════════════════════════════════════

SCALES = {
    'Chromatic': list(range(12)),
    'Pentatonic': [0, 2, 4, 7, 9],
    'Major': [0, 2, 4, 5, 7, 9, 11],
    'Minor': [0, 2, 3, 5, 7, 8, 10],
    'Microtonal': [0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 5, 5.5, 6, 7, 7.5, 8, 9, 9.5, 10, 11, 11.5],
    'Harmonic': [0, 2, 4, 7, 12, 14, 16, 19, 24],
}


def quantize_to_scale(freq: float, tonic: float, scale_name: str) -> float:
    """Snap a frequency to the nearest scale degree relative to tonic."""
    if scale_name not in SCALES or scale_name == 'Chromatic':
        return freq  # Chromatic = no quantization
    scale_degrees = SCALES[scale_name]
    if freq <= 0 or tonic <= 0:
        return freq
    semitones_from_tonic = 12.0 * np.log2(freq / tonic)
    octave = np.floor(semitones_from_tonic / 12.0)
    semitone_in_octave = semitones_from_tonic - octave * 12.0
    best_dist = 12.0
    best_degree = 0.0
    for sd in scale_degrees:
        for shift in [-12, 0, 12]:
            candidate = sd + shift
            dist = abs(semitone_in_octave - candidate)
            if dist < best_dist:
                best_dist = dist
                best_degree = candidate
    quantized_semitones = octave * 12.0 + best_degree
    return tonic * (2.0 ** (quantized_semitones / 12.0))


def map_tonic_spread_to_freq(state: np.ndarray, tonic_norm: float,
                              scale_name: str, pitch_spread: float) -> float:
    """Map chaos state to frequency using tonic and pitch spread controls."""
    tonic_hz = 27.5 * (2.0 ** (tonic_norm * 7.25))  # A0 (~27.5) to ~4200 Hz
    spread_octaves = 0.5 + pitch_spread * 4.5  # 0.5 to 5.0 octaves
    deviation = (float(state[0]) - 0.5) * 2.0 * spread_octaves
    freq = tonic_hz * (2.0 ** deviation)
    return quantize_to_scale(freq, tonic_hz, scale_name)


def map_dynamic_to_amp(state: np.ndarray, dynamic_norm: float) -> float:
    """Map chaos state to amplitude using dynamic range control."""
    amp_base = 0.01 + dynamic_norm * 0.49  # 0.01 to 0.5
    range_factor = 0.3 + dynamic_norm * 9.7  # 0.3 to 10.0
    deviation = (float(state[1]) - 0.5) * range_factor
    return float(np.clip(amp_base + deviation * 0.05, 0.001, 1.0))


# ═══════════════════════════════════════════════════════════════════════
# Generative Layer — Cellular Automata + L-System (Phase 2)
# ═══════════════════════════════════════════════════════════════════════

class CellularAutomaton:
    """1D cellular automaton. Rule 30 or 110 for rhythm triggering."""

    def __init__(self, width: int = 64, rule: int = 30):
        self.width = width
        self.rule = rule
        self.cells = np.zeros(width, dtype=np.uint8)
        self.cells[width // 2] = 1  # single seed in middle
        self.step_count = 0

    def step(self) -> np.ndarray:
        """Advance one generation. Returns current row as uint8 array."""
        new_cells = np.zeros(self.width, dtype=np.uint8)
        for i in range(self.width):
            left = self.cells[(i - 1) % self.width]
            center = self.cells[i]
            right = self.cells[(i + 1) % self.width]
            pattern = (left << 2) | (center << 1) | right
            new_cells[i] = (self.rule >> pattern) & 1
        self.cells = new_cells
        self.step_count += 1
        return self.cells

    def trigger_pattern(self) -> list:
        """Return list of 8 bools from current row for voice triggering."""
        # Sample 8 evenly-spaced positions
        step = max(1, self.width // 8)
        return [bool(self.cells[i * step]) for i in range(8)]


class LSystem:
    """L-System for melody generation."""

    def __init__(self, axiom: str = "A", rules: dict = None, angle: float = 60.0):
        self.axiom = axiom
        self.rules = rules or {"A": "AB", "B": "A"}  # Fibonacci L-system
        self.string = axiom
        self.generation = 0

    def iterate(self, max_len: int = 200) -> str:
        """Apply production rules. Cap string at max_len."""
        result = []
        for ch in self.string[:max_len]:
            result.append(self.rules.get(ch, ch))
            if len(result) >= max_len:
                break
        self.string = "".join(result)[:max_len]
        self.generation += 1
        return self.string

    def to_notes(self, base_freq: float = 220.0, scale: list = None) -> list:
        """Convert L-system string to MIDI-style frequency list.
        Maps characters to scale degrees."""
        if scale is None:
            scale = [0, 2, 4, 5, 7, 9, 11]  # major scale degrees
        notes = []
        n = len(self.string)
        for i, ch in enumerate(self.string):
            if ch in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                degree = (ord(ch) - ord('A')) % len(scale)
                octave = 4 + (i % 3)  # spread across 3 octaves
                freq = base_freq * (2.0 ** ((octave - 4) + scale[degree] / 12.0))
                notes.append(freq)
        return notes if notes else [440.0]


# ═══════════════════════════════════════════════════════════════════════

# 10. Entry point — audio callback + feedback + coupling
# ═══════════════════════════════════════════════════════════════════════

def run(duration=None, device=None, verbose=False,
        tonic=0.5, scale="Pentatonic", pitch_spread=0.3, dynamic=0.5):
    """Launch the synth. duration=None runs forever (Ctrl+C to stop).
    
    Performance params (all 0.0-1.0 normalized):
      tonic: root pitch (0=A0/27.5Hz, 1=~4.2kHz)
      scale: one of 'Chromatic','Pentatonic','Major','Minor','Microtonal','Harmonic'
      pitch_spread: octave range (0=0.5 oct, 1=5.0 oct)
      dynamic: amplitude range (0=quiet/flat, 1=loud/wide)
    """
    import sounddevice as sd

    chaos = LorenzAttractor()
    manifold = ManifoldMapper(n_centroids=16)
    pool = VoicePool(capacity=128, max_active=16)
    coupling = CouplingField()
    ca = CellularAutomaton(width=64, rule=30)
    lsys = LSystem(axiom="ABCB", rules={"A": "ABC", "B": "BAB", "C": "CA"})
    lsys_callback_counter = 0
    ca_callback_counter = 0

    # Phase 3: new engine components
    ssb = SelfSampleBuffer(duration_s=2.0)
    delay_net = DelayNetwork()
    ltfb = LongTermFeedback()
    ltfb_frame_counter = 0  # tick every ~172 frames (1 second)

    trigger_gates = [True] * 8  # initial: all gates open
    feedback_state = {
        'centroid_history': [],
        'flux_history': [],
        'zcr_history': [],
        'silence_counter': 0,
    }

    # Audio callback as closure — captures all state
    def audio_callback(outdata, frames, time_info, status):
        outdata.fill(0.0)

        nonlocal ca_callback_counter, lsys_callback_counter, trigger_gates, ltfb_frame_counter

        # 0. Generative layer: CA rhythm + L-System melody
        ca_callback_counter += 1
        ca_step_interval = 4  # step CA every 4 callbacks (~23ms)
        if ca_callback_counter % ca_step_interval == 0:
            ca.step()
            trigger_gates = ca.trigger_pattern()  # 8 bools

        lsys_callback_counter += 1
        lsys_step_interval = 20  # iterate L-system every 20 callbacks (~116ms)
        if lsys_callback_counter % lsys_step_interval == 0:
            lsys.iterate()
            ca.step()  # CA evolves with L-system
        melody_notes = lsys.to_notes(base_freq=110.0) if lsys_callback_counter % lsys_step_interval == 0 else []

        # 1. Step chaos engine -> 3D state
        state = chaos.step()
        eid, bid, mid = manifold.find_nearest(state)

        # Use CA gates: always trigger, but gate=1 → full amp, gate=0 → half amp
        gate_idx = int(state[2] * 8) % 8
        freq = map_tonic_spread_to_freq(state, tonic, scale, pitch_spread)
        if melody_notes and np.random.rand() < 0.2:
            freq = melody_notes[np.random.randint(0, len(melody_notes))]
        amp = map_dynamic_to_amp(state, dynamic)
        if not trigger_gates[gate_idx]:
            amp *= 0.3  # gated voices are quieter but still present
        pool.trigger(eid, bid, mid, freq, amp, float(state[2]))
        pool.render(outdata.T)

        # Phase 3a: write stereo to self-sample ring buffer
        ssb.write(outdata.T.copy())

        # 2. Multi-feature extraction
        mono = outdata.mean(axis=1)
        rms = np.sqrt(np.mean(mono ** 2))
        centroid = compute_spectral_centroid(mono, SAMPLE_RATE)
        zcr = float(np.sum(np.abs(np.diff(np.sign(mono))))
                    / (2.0 * len(mono))) if len(mono) > 0 else 0.0
        prev = feedback_state['centroid_history'][-1] if feedback_state['centroid_history'] else centroid
        flux = abs(centroid - prev)

        # Phase 3b: Long-term feedback — feed every frame, tick every ~1s
        ltfb.feed(centroid, flux, rms)
        ltfb_frame_counter += 1
        if ltfb_frame_counter >= ltfb.window_size:
            ltfb_frame_counter = 0
            lt_mod = ltfb.tick()
            if lt_mod:
                # lt_brightness (0=dark,1=bright) → rho (Lorenz attractor parameter)
                if hasattr(chaos, 'rho'):
                    target_rho = 15.0 + lt_mod.get('lt_brightness', 0.5) * 25.0
                    chaos.rho += (target_rho - chaos.rho) * 0.02
                # lt_activity (0=calm,1=busy) → dt (step size)
                if hasattr(chaos, 'dt'):
                    activity = lt_mod.get('lt_activity', 0.5)
                    target_dt = 0.005 + activity * 0.04
                    chaos.dt += (target_dt - chaos.dt) * 0.02

        # 3. Smooth histories (10-frame windows)
        feedback_state['centroid_history'].append(centroid)
        feedback_state['flux_history'].append(flux)
        feedback_state['zcr_history'].append(zcr)
        for h in ['centroid_history', 'flux_history', 'zcr_history']:
            if len(feedback_state[h]) > 10:
                feedback_state[h].pop(0)

        avg_centroid = np.mean(feedback_state['centroid_history'])
        avg_flux = np.mean(feedback_state['flux_history'])
        avg_zcr = np.mean(feedback_state['zcr_history'])

        # 4. Feedback: centroid -> Lorenz sigma (orbit shape)
        if hasattr(chaos, 'sigma'):
            target_sigma = 6.0 + (1.0 - min(avg_centroid / 5000.0, 1.0)) * 9.0
            chaos.sigma += (target_sigma - chaos.sigma) * 0.01

        # 5. Feedback: spectral flux -> dt (step size)
        if hasattr(chaos, 'dt'):
            target_dt = 0.005 + (1.0 - min(avg_flux / 1000.0, 1.0)) * 0.02
            chaos.dt += (target_dt - chaos.dt) * 0.01

        # 6. Feedback: ZCR -> active voice count
        target_active = int(8 + (1.0 - min(avg_zcr, 1.0)) * 16)
        pool.max_active = max(4, min(32, target_active))

        # 7. Coupling field: deposit RMS, read back as extra gain
        coupling.deposit(rms * 2.0)
        coupling.tick()
        extra = coupling.read()
        outdata[:] *= (1.0 + extra * 0.3)

        # Phase 3c: process through delay network (cross-feedback lines)
        outdata[:] = delay_net.process(outdata.T).T

        # 8. Cold start: inject noise after 500ms silence
        feedback_state['silence_counter'] += frames
        if rms > 1e-4:
            feedback_state['silence_counter'] = 0
        if feedback_state['silence_counter'] > int(SAMPLE_RATE * 0.5):
            noise = np.random.randn(frames).astype(np.float32) * 0.01
            outdata[:, 0] += noise
            outdata[:, 1] += noise
            feedback_state['silence_counter'] = 0

        # 9. Slow attractor parameter drift
        if hasattr(chaos, 'sigma'):
            if 'sigma_drift' not in feedback_state:
                feedback_state['sigma_drift'] = 0.0
            feedback_state['sigma_drift'] += np.random.randn() * 0.002
            feedback_state['sigma_drift'] = np.clip(feedback_state['sigma_drift'], -1.0, 1.0)
            chaos.sigma = np.clip(10.0 + feedback_state['sigma_drift'], 6.0, 15.0)

    print("chaos-synth v0.3.0 [Phase 3 — Performance Handles + Full Integration]")
    print(f"  Chaos: {type(chaos).__name__}")
    print(f"  Manifold: {len(manifold.centroids)} centroids on [0,1]^3")
    print(f"  Pool: {pool.capacity} slots, max {pool.max_active} active")
    print(f"  Coupling: ON, 44100-sample ring buffer")
    print(f"  DelayNetwork: 4-line cross-feedback (150/225/337/506ms)")
    print(f"  LongTermFeedback: 1s spectral averaging -> rho+dt")
    print(f"  SelfSampleBuffer: 2s ring buffer")
    print(f"  Performance: tonic={tonic:.2f} scale={scale} spread={pitch_spread:.2f} dyn={dynamic:.2f}")
    print(f"  CA: Rule 30, 64 cells | LS: 3 rules, axiom=ABCB")
    print(f"  {SAMPLE_RATE}Hz / block {BLOCK_SIZE} / Ctrl+C to stop")

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
                    if verbose:
                        stats = f"\r  voices:{pool._active_count()}/{pool.max_active} sigma:{chaos.sigma:.1f} rc:{coupling.read():.3f}  "
                        print(stats, end='', flush=True)
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    run()
