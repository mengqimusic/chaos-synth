# Phase 3 — Deep Feedback Engine + Real-Time Performance UI

> **For Hermes:** Implement task-by-task. Each task verified before next.

**Goal:** Upgrade synthesis engine with self-sampling, delay network, long-term feedback, and presets; then build a real-time performance UI with tactile macro controls and live spectrum visualization.

**Architecture:** All engine upgrades stay in `src/main.py` (keep single-file core). Delay network and self-sampling are new classes. UI is a separate entry point `src/ui/app.py` using dearpygui for immediate-mode real-time controls + live FFT plot.

**Tech Stack:** Python 3.9, numpy, sounddevice, dearpygui, scipy, TOML

---

## 设计理念

> 好的声音 + 好的反应 = 好的乐器

Phase 3 完成合成引擎的最后一块拼图（自采样），然后建立一套 **9 控件演奏界面**：
- **4 个演奏手柄**（直接控制音高和动态边界）：Tonic, Scale, Dynamic, Pitch Spread
- **5 个生态塑造**（混沌调制）：Material, Density, Mutation, Coherence, Feedback

设计哲学：**混沌做变化，你定边界。** 就像拉小提琴——你控制音高和力度，琴体和弦的混沌交互产生丰富的泛音和微变化。转一个生态旋钮，整个声音生态就会发生质变。

UI 核心理念：**不是调参数，是玩生态。**

---

## Task Group A: Phase 3 Engine Upgrades

### Task A1: Self-Sampling Ring Buffer + Transient Snatch Exciter

**Objective:** 实现自采样环形缓冲区，让 exciter #11 (Transient Snatch) 从系统自身输出中抓取瞬态片段作为激发源——实现"活体吞噬"。

**Files:**
- Modify: `src/main.py` — add `SelfSampleBuffer` class, update `exciter_transient()`

**Implementation:**

```python
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
            seg = self.buffer.mean(axis=0)[start:start+length].copy()
        else:
            seg = np.concatenate([
                self.buffer.mean(axis=0)[start:],
                self.buffer.mean(axis=0)[:length - (self.length - start)]
            ])
        # Apply hanning window to smooth edges
        return seg * np.hanning(length).astype(np.float32)
```

Update `exciter_transient()` to accept a `self_sample` parameter:
```python
def exciter_transient(freq: float, sr: int = SAMPLE_RATE,
                      self_sample: np.ndarray = None) -> np.ndarray:
    """#11: Transient snatch from self-sampling ring buffer."""
    if self_sample is not None and len(self_sample) > 0:
        # Use the self-sampled audio as the grain
        return self_sample.astype(np.float32) * 0.8
    # Fallback: filtered noise burst
    buf = np.random.randn(128).astype(np.float32) * 0.2 * np.hanning(128).astype(np.float32)
    return buf
```

**Verification:** Render 5s offline WAV with self-sampling enabled. Exciter #11 should produce audible snippets of previous output instead of noise bursts.

---

### Task A2: Delay Network (No-Input Mixer Style)

**Objective:** 4 条并行延迟线，带交叉反馈矩阵，模拟 no-input mixer 的自激振荡行为。

**Files:**
- Modify: `src/main.py` — add `DelayNetwork` class

**Implementation:**

```python
class DelayNetwork:
    """4 parallel delay lines with cross-feedback matrix."""
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
```

**Verification:** Render 5s offline WAV with delay network enabled. Should hear distinct echo trails with cross-feedback interactions.

---

### Task A3: Long-Term Feedback Path

**Objective:** 在现有帧级反馈之外，添加秒级长时反馈——1 秒窗口频谱特征缓慢回控混沌引擎参数，产生"呼吸感"。

**Files:**
- Modify: `src/main.py` — add `LongTermFeedback` class, integrate into callback/render loop

**Implementation:**

```python
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
```

Integrate into callback: every ~172 frames, apply long-term modulation to:
- Lorenz `rho` (orbit size, affected by brightness)
- Lorenz `dt` (step speed, affected by activity)
- Manifold Mutation macro (affected by activity)

**Verification:** Run 30s render. Sound should evolve over seconds, not just frame-to-frame. Spectral brightness should slowly oscillate.

---

### Task A4: Preset System (Save/Load/Morph)

**Objective:** TOML 预设系统：保存/加载参数快照，支持两个预设之间的 morphing 渐变。

**Files:**
- Create: `src/presets.py` — PresetManager class
- Create: `config/presets/` — directory for preset files

**Implementation:**

```python
# src/presets.py
import toml
import os

PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'presets')

class PresetManager:
    """Save/load/morph parameter snapshots."""
    
    @staticmethod
    def capture(chaos, manifold, pool, coupling, delay_net, macros: dict) -> dict:
        """Capture current state as a preset dict."""
        preset = {
            'name': 'untitled',
            'chaos': {
                'type': type(chaos).__name__,
                'sigma': getattr(chaos, 'sigma', 10.0),
                'rho': getattr(chaos, 'rho', 28.0),
                'beta': getattr(chaos, 'beta', 2.667),
                'dt': getattr(chaos, 'dt', 0.01),
            },
            'manifold': {
                'n_centroids': len(manifold.centroids),
            },
            'pool': {
                'capacity': pool.capacity,
                'max_active': pool.max_active,
            },
            'macros': macros,  # Material, Density, Mutation, Coherence, Feedback, Energy
        }
        return preset
    
    @staticmethod
    def save(preset: dict, name: str):
        """Save preset to config/presets/<name>.toml."""
        os.makedirs(PRESET_DIR, exist_ok=True)
        path = os.path.join(PRESET_DIR, f'{name}.toml')
        with open(path, 'w') as f:
            toml.dump(preset, f)
    
    @staticmethod
    def load(name: str) -> dict:
        """Load preset from config/presets/<name>.toml."""
        path = os.path.join(PRESET_DIR, f'{name}.toml')
        with open(path, 'r') as f:
            return toml.load(f)
    
    @staticmethod
    def list_presets() -> list:
        """List available preset names."""
        if not os.path.isdir(PRESET_DIR):
            return []
        return [f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith('.toml')]
    
    @staticmethod
    def morph(preset_a: dict, preset_b: dict, t: float) -> dict:
        """Linear interpolation between two presets. t=0→A, t=1→B."""
        result = {}
        for key in preset_a:
            if key == 'name':
                result[key] = f'morph_{preset_a["name"]}_{preset_b["name"]}'
            elif isinstance(preset_a[key], dict):
                result[key] = {}
                for subkey in preset_a[key]:
                    if subkey in preset_b.get(key, {}):
                        a_val = preset_a[key][subkey]
                        b_val = preset_b[key][subkey]
                        if isinstance(a_val, (int, float)):
                            result[key][subkey] = a_val + (b_val - a_val) * t
                        else:
                            result[key][subkey] = a_val if t < 0.5 else b_val
        return result
```

**Verification:** Create a preset, save it, load it back, verify parameters match. Morph between two presets at t=0.5.

---

### Task A5: Performance Handle Mapping Functions

**Objective:** 实现 Tonic/Scale/Dynamic/Pitch Spread 四个演奏手柄的参数映射函数，替换现有的 `map_state_to_freq()` 和 `map_state_to_amp()`。

**Files:**
- Modify: `src/main.py` — add `quantize_to_scale()`, `map_tonic_spread_to_freq()`, `map_dynamic_to_amp()`

**Implementation:**

```python
# Scale degree definitions (semitones from tonic)
SCALES = {
    'Chromatic':  list(range(12)),                    # all 12 semitones
    'Pentatonic': [0, 2, 4, 7, 9],                    # major pentatonic
    'Major':      [0, 2, 4, 5, 7, 9, 11],             # major scale
    'Minor':      [0, 2, 3, 5, 7, 8, 10],             # natural minor
    'Microtonal': [0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 5, 5.5, 6, 7, 7.5, 8, 9, 9.5, 10, 11, 11.5],
    'Harmonic':   [0, 2, 4, 7, 12, 14, 16, 19, 24],  # harmonic series intervals
}

def quantize_to_scale(freq: float, tonic: float, scale_name: str) -> float:
    """Quantize a continuous frequency to the nearest scale degree."""
    if scale_name not in SCALES or scale_name == 'Chromatic':
        return freq  # Chromatic = no quantization
    
    degrees = SCALES[scale_name]
    # Find which octave the freq is in relative to tonic
    ratio = freq / max(tonic, 1e-6)
    octave = np.floor(np.log2(ratio))
    semitones = 12.0 * np.log2(ratio) - octave * 12.0
    
    # Find nearest scale degree
    best_degree = min(degrees, key=lambda d: abs(semitones - d))
    return tonic * (2.0 ** (octave + best_degree / 12.0))

def map_tonic_spread_to_freq(state: np.ndarray, tonic_norm: float,
                              scale_name: str, pitch_spread: float) -> float:
    """Map chaos state to frequency with tonic center and spread.
    
    tonic_norm: 0-1 → 27.5-4186 Hz
    pitch_spread: 0-1 → ±0.5 to ±5 octave spread
    scale_name: quantizes output to scale (unless Chromatic)
    """
    tonic_hz = 27.5 * (2.0 ** (tonic_norm * 7.25))  # 27.5 - 4186 Hz
    spread_octaves = 0.5 + pitch_spread * 4.5  # 0.5 - 5.0 octaves
    # state[0] in [0,1] → deviation from tonic
    deviation = (float(state[0]) - 0.5) * 2.0 * spread_octaves
    freq = tonic_hz * (2.0 ** deviation)
    return quantize_to_scale(freq, tonic_hz, scale_name)

def map_dynamic_to_amp(state: np.ndarray, dynamic_norm: float) -> float:
    """Map chaos state + dynamic knob to amplitude.
    
    dynamic_norm: 0-1
      0.0 → amp_base=0.01, narrow range (whisper)
      0.5 → amp_base=0.15, medium range
      1.0 → amp_base=0.5,  wide range (roar)
    """
    amp_base = 0.01 + dynamic_norm * 0.49  # 0.01 - 0.5
    range_factor = 0.3 + dynamic_norm * 9.7  # 0.3 - 10.0
    # state[1] modulates around amp_base
    deviation = (float(state[1]) - 0.5) * range_factor
    return np.clip(amp_base + deviation * 0.05, 0.001, 1.0)
```

Update `render_wav.py` and `run()` callback to use these new mapping functions instead of `map_state_to_freq()` and `map_state_to_amp()`.

**Verification:** Test with different tonic/spread/scale values. Frequencies should center around tonic, spread should control deviation range, and scale should quantize output.


### Task A6: Integrate All Phase 3 Engine Features into run() and render_wav.py

**Objective:** 将 SelfSampleBuffer, DelayNetwork, LongTermFeedback, PresetManager 全部接入主循环和离线渲染。

**Files:**
- Modify: `src/main.py` — update `run()` callback to use all new components
- Modify: `render_wav.py` — mirror all changes

**Verification:** Render 10s offline WAV with all Phase 3 features active. Sound should be richer, with audible self-sampling artifacts, delay trails, and slow spectral evolution.

---

## Task Group B: Real-Time Performance UI

### Task B1: Install dearpygui and Verify

**Objective:** Install `dearpygui` package in the project venv.

```bash
cd ~/Documents/Code/chaos-synth && .venv/bin/pip install dearpygui
```

**Verification:** Run `python -c "import dearpygui.dearpygui as dpg; print(dpg.get_dearpygui_version())"` — should print version string.

---

### Task B2: UI App Skeleton — Window + Audio Thread

**Objective:** 创建 `src/ui/app.py`，建立 dearpygui 窗口 + 后台音频线程，实现低延迟实时演奏的基础框架。

**Files:**
- Create: `src/ui/app.py`

**Architecture:**
- 主线程: dearpygui 渲染循环 (~60fps)
- 音频线程: sounddevice OutputStream callback (独立 C 线程)
- 共享状态: `threading.Lock` 保护的参数字典，UI 写入，音频线程读取
- 频谱数据: callback 每帧 push FFT 数据到线程安全队列，UI 线程消费并绘图

```python
# src/ui/app.py
import threading
import queue
import numpy as np
import dearpygui.dearpygui as dpg
import sounddevice as sd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

SAMPLE_RATE = 44100
BLOCK_SIZE = 256

class ChaosSynthUI:
    def __init__(self):
        # Shared state between UI thread and audio thread
        self.lock = threading.Lock()
        self.params = {
            # Performance handles (direct control)
            'tonic': 0.5,           # base frequency 27.5-4186Hz
            'scale': 'Pentatonic',  # scale constraint
            'dynamic': 0.5,         # volume + dynamic range (linked)
            'pitch_spread': 0.3,    # pitch emergence amplitude
            # Ecosystem shapers (chaos modulation)
            'material': 0.5,        # timbre bias (transient <-> resonant)
            'density': 0.5,         # voice density 4-32
            'mutation': 0.3,        # chaos variation speed
            'coherence': 0.7,       # fragment <-> fusion
            'feedback': 0.5,        # self-excitation amount
        }
        self.spectrum_queue = queue.Queue(maxsize=10)
        self.running = True
        
    def audio_callback(self, outdata, frames, time_info, status):
        """Called from sounddevice C thread. Reads UI params, generates audio."""
        outdata.fill(0.0)
        with self.lock:
            p = dict(self.params)  # snapshot current params
        
        # TODO: wire up actual synth engine with p['tonic'] etc
        # For now: generate test tone reflecting all 4 performance handles
        t = np.arange(frames, dtype=np.float32) / SAMPLE_RATE
        base_freq = 27.5 * (2.0 ** (p['tonic'] * 7.25))  # 27.5 - 4186 Hz
        freq = base_freq + p['pitch_spread'] * 880.0 * np.sin(t * 5.0)[0]
        amp = 0.01 + p['dynamic'] * 0.2
        tone = np.sin(2.0 * np.pi * freq * t) * amp
        outdata[:, 0] = tone
        outdata[:, 1] = tone
        
        # Push spectrum for UI
        if not self.spectrum_queue.full():
            spectrum = np.abs(np.fft.rfft(tone))
            self.spectrum_queue.put_nowait(spectrum[:128])
    
    def start_audio(self):
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            channels=2, dtype='float32', callback=self.audio_callback
        )
        self.stream.start()
    
    def build_ui(self):
        dpg.create_context()
        
        with dpg.window(label="Chaos Synth", width=800, height=600):
            # --- Performance Handles (top row, 4 controls) ---
            dpg.add_text("PERFORMANCE", color=(255, 200, 100))
            with dpg.group(horizontal=True):
                dpg.add_slider_float(tag="tonic", label="Tonic",
                    default_value=0.5, min_value=0.0, max_value=1.0,
                    callback=self.on_param_change, user_data="tonic",
                    width=100, height=120, vertical=True)
                dpg.add_combo(tag="scale", label="Scale",
                    items=["Chromatic","Pentatonic","Major","Minor","Microtonal","Harmonic"],
                    default_value="Pentatonic", width=100,
                    callback=self.on_param_change, user_data="scale")
                dpg.add_slider_float(tag="dynamic", label="Dynamic",
                    default_value=0.5, min_value=0.0, max_value=1.0,
                    callback=self.on_param_change, user_data="dynamic",
                    width=100, height=120, vertical=True)
                dpg.add_slider_float(tag="pitch_spread", label="Spread",
                    default_value=0.3, min_value=0.0, max_value=1.0,
                    callback=self.on_param_change, user_data="pitch_spread",
                    width=100, height=120, vertical=True)

            # --- Ecosystem Shapers (bottom row, 5 controls) ---
            dpg.add_text("ECOSYSTEM", color=(100, 200, 255))
            with dpg.group(horizontal=True):
                for name in ['material', 'density', 'mutation', 'coherence', 'feedback']:
                    with dpg.group():
                        dpg.add_text(name.upper())
                        dpg.add_slider_float(
                            tag=name, default_value=0.5,
                            min_value=0.0, max_value=1.0,
                            callback=self.on_param_change, user_data=name,
                            width=100, height=120, vertical=True
                        )
            
            # Spectrum display
            with dpg.plot(label="Spectrum", height=200, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, label="Freq")
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Mag", tag="y_axis")
                self.spectrum_series = dpg.add_line_series(
                    list(range(128)), [0]*128, parent=y_axis, tag="spectrum_line"
                )
        
        dpg.create_viewport(title="Chaos Synth", width=820, height=640)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def on_param_change(self, sender, app_data, user_data):
        """Called when any slider moves."""
        with self.lock:
            self.params[user_data] = app_data
    
    def update_spectrum(self):
        """Pull latest spectrum from queue and update plot."""
        try:
            spec = self.spectrum_queue.get_nowait()
            dpg.set_value("spectrum_line", [list(range(len(spec))), list(spec)])
        except queue.Empty:
            pass
    
    def run(self):
        self.build_ui()
        self.start_audio()
        while dpg.is_dearpygui_running() and self.running:
            self.update_spectrum()
            dpg.render_dearpygui_frame()
        self.stream.stop()
        dpg.destroy_context()

if __name__ == '__main__':
    ui = ChaosSynthUI()
    ui.run()
```

**Verification:** Run `python src/ui/app.py`. Should see window with 9 controls in two labeled rows (PERFORMANCE: Tonic slider + Scale combo + Dynamic slider + Spread slider; ECOSYSTEM: 5 sliders) and a spectrum plot. Moving sliders should change tone. Audio should be continuous. Close window to exit.

---

### Task B3: Wire Real Synth Engine into UI

**Objective:** 用实际的 Chaos Synth 引擎替换 UI 中的测试音，将 9 个控件映射到混沌参数。

**Files:**
- Modify: `src/ui/app.py` — import and wire `LorenzAttractor`, `ManifoldMapper`, `VoicePool`, `CouplingField`, `CellularAutomaton`, `LSystem`, `DelayNetwork`, `SelfSampleBuffer`, `LongTermFeedback`

**Performance Handles（演奏手柄）:**

| # | 控件 | 范围 | 映射到 | 效果 |
|---|------|------|--------|------|
| 1 | **Tonic** | 0-1 | `base_freq = 27.5 * 2^(v * 7.25)` → 27.5–4186Hz | 基准音高中心 |
| 2 | **Scale** | 6 选项 | 音阶约束网格（Chromatic/Pentatonic/Major/Minor/Microtonal/Harmonic） | 音高量化到音阶内 |
| 3 | **Dynamic** | 0-1 | 基础音量 + 动态范围联动: `amp_base = 0.01+v*0.3`, `compressor_ratio = 1+v*19` | 耳语↔咆哮 |
| 4 | **Pitch Spread** | 0-1 | `state[0]` 映射范围：`spread=v*5.0`，频率 `= tonic * 2^(spread * (state-0.5))` | 微漂移↔全范围跳跃 |

**Ecosystem Shapers（生态塑造）:**

| # | 控件 | 范围 | 映射到 | 效果 |
|---|------|------|--------|------|
| 5 | **Material** | 0-1 | Voronoi 质心偏移 + exciter 偏好 (0=瞬态, 1=共振) | 打击↔持续 |
| 6 | **Density** | 0-1 | 触发密度 + voice count `4+v*28` + CA step speed | 稀疏↔密集 |
| 7 | **Mutation** | 0-1 | Lorenz dt 偏移 + 质心抖动幅度 | 稳定↔疯狂 |
| 8 | **Coherence** | 0-1 | 反馈平滑系数 + 耦合场衰减速度 | 碎片↔融合 |
| 9 | **Feedback** | 0-1 | 自激反馈量 + 延迟网络 wet mix | 干声↔淹没 |

**Verification:** Run `python src/ui/app.py`. All 9 controls should visibly and audibly change the sound. Spectrum should show real-time FFT.

---

### Task B4: Add Preset Controls to UI

**Objective:** 在 UI 中添加预设保存/加载按钮，支持 morphing 滑块。

**Files:**
- Modify: `src/ui/app.py` — add preset panel

**Implementation:**
```python
# Add to build_ui():
with dpg.group(horizontal=True):
    dpg.add_input_text(tag="preset_name", default_value="my_preset", width=120)
    dpg.add_button(label="Save", callback=self.save_preset)
    dpg.add_button(label="Load", callback=self.load_preset)
    dpg.add_combo(tag="preset_list", items=[], width=150, callback=self.on_preset_select)

# Morph slider
dpg.add_slider_float(tag="morph_amount", label="Morph A→B", 
                     default_value=0.0, min_value=0.0, max_value=1.0,
                     callback=self.on_morph)
```

**Verification:** Create 2 presets with different parameter settings. Morph between them. Sound should smoothly transition.

---

### Task B5: Attractor Visualization

**Objective:** 在 UI 中添加 Lorenz 吸引子的 2D 投影实时可视化 (XY, XZ, YZ 三个子图)。

**Files:**
- Modify: `src/ui/app.py` — add attractor plot

**Implementation:**
- 音频线程每次 callback 后将当前 attractor 坐标 push 到队列
- UI 线程维护一个 200 点的轨迹缓冲区
- 用 dearpygui 的 `draw_line` 绘制轨迹
- 用 `draw_circle` 标记当前位置

**Verification:** Run UI. Should see Lorenz attractor trajectory updating in real-time. Shape should visibly change when Mutation or Coherence sliders are moved.

---

## 验收标准

1. **Phase 3 引擎**：离线渲染 10s WAV，能听到自采样回响、延迟网络尾巴、缓慢的频谱演化
2. **UI 响应**：9 个控件实时影响声音，延迟 < 50ms
3. **预设**：保存 2 个截然不同的预设，morph 过渡顺畅
4. **可视化**：频谱 + 吸引子轨迹实时更新，不卡顿
5. **稳定性**：连续运行 5 分钟不崩溃，CPU < 30%
