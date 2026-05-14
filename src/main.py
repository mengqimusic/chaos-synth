# chaos-synth — Granular Synthesis Engine
# 12 exciters × 10 bodies × 7 modulators = 2520 voice combos
# VoicePool + SelfSampleBuffer + DelayNetwork
# Pure numpy, float32, single file.
# Pure numpy, float32, single file.

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
BLOCK_SIZE = 256

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

            pan_raw = v.get('pan', v['chaos_x'])
            pan = pan_raw * 2.0 - 1.0
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



# ═══════════════════════════════════════════════════════════════════════

class TimbreMap:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        all_combos = [(e,b,m) for e in range(12) for b in range(10) for m in range(7)]
        self.all_combos = all_combos
        self.maps = self._build_maps()
        self.current_map = "Gradual"
        self._arrangement = self.maps[self.current_map]

    def _build_maps(self):
        maps = {}
        all_c = self.all_combos
        maps["Gradual"] = sorted(all_c, key=lambda c: (c[0], c[1], c[2]))
        maps["Dirty"] = sorted(all_c, key=lambda c: (-c[0], c[1], c[2]))
        maps["Clean→Broken"] = sorted(all_c, key=lambda c: (c[1], c[0], c[2]))
        body_order = {0:0,2:1,4:2,5:3, 1:4,3:5,6:6,7:7,8:8,9:9}
        maps["Strike→Sing"] = sorted(all_c, key=lambda c: (body_order[c[1]], c[0], c[2]))
        exc_tuned = {0:0,4:1,6:2,2:3,5:4,8:5,7:6,1:7,10:8,9:9,3:10,11:11}
        maps["Tuned→Noise"] = sorted(all_c, key=lambda c: (exc_tuned[c[0]], c[1], c[2]))
        maps["Tight→Vast"] = sorted(all_c, key=lambda c: (c[2], c[1], c[0]))
        # Morton Z-order
        def morton_key(c):
            def p1b2(n):
                n &= 0x3ff; n = (n|(n<<16))&0x30000ff; n = (n|(n<<8))&0x300f00f
                n = (n|(n<<4))&0x30c30c3; n = (n|(n<<2))&0x9249249; return n
            return p1b2(c[0])|(p1b2(c[1])<<1)|(p1b2(c[2])<<2)
        maps["Full Jump"] = sorted(all_c, key=morton_key)
        rng = np.random.RandomState(42)
        shuffled = all_c.copy(); rng.shuffle(shuffled)
        maps["Random"] = shuffled
        return maps

    def set_map(self, name):
        if name in self.maps:
            self.current_map = name
            self._arrangement = self.maps[name]

    def get_combo(self, position):
        idx = int(np.clip(position,0,1) * (len(self._arrangement)-1))
        return self._arrangement[idx]

    @property
    def map_names(self):
        return list(self.maps.keys())


# ═══════════════════════════════════════════════════════════════════════

class LFO:
    WAVEFORMS = ["Sine","Triangle","Square","Saw","Random"]
    def __init__(self, name="LFO1"):
        self.name = name; self.waveform="Sine"; self.rate=1.0
        self.depth=0.0; self.target="Pitch"
        self._phase=0.0; self._last_random=0.0; self._value=0.0

    def tick(self, dt):
        self._phase += self.rate * dt
        self._phase %= 1.0
        self._value = self._sample(self._phase)
        return self._value

    def _sample(self, phase):
        if self.waveform=="Sine": return float(np.sin(2*np.pi*phase))
        elif self.waveform=="Triangle": return float(4*abs(phase-0.5)-1)
        elif self.waveform=="Square": return 1.0 if phase<0.5 else -1.0
        elif self.waveform=="Saw": return float(2*phase-1)
        elif self.waveform=="Random":
            if phase < 0.01: self._last_random = float(np.random.uniform(-1,1))
            return self._last_random
        return 0.0

    @property
    def value(self):
        return self._value if self.depth>0.001 and self.rate>0.001 else 0.0
