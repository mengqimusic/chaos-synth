# Phase 4 — 经典粒子合成重构：手动控制 + LFO + Timbre Map

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 去掉混沌引擎、Voronoi、CA/L-System、FFT 反馈分析等所有间接层，改为经典粒子合成架构：手动参数直接控制粒子触发，LFO 调制，2520 combo 通过 Timbre Map 排列选取。

**Architecture:** `src/main.py` 保留 exciter/body/modulator 库 + VoicePool + SelfSampleBuffer + DelayNetwork，新增 `TimbreMap` 和 `LFO`。`src/ui/app.py` 重写为简洁的 8 参数 + 3 LFO 界面。移除约 500 行废弃代码。

**Tech Stack:** Python 3.9, numpy, sounddevice, dearpygui, TOML

---

## 设计

### 控件布局

```
PARTICLE:  Rate | Pitch | Size | Feedback
SPREAD:    Pitch Spread | Pan Spread | Pos Spread
TIMBRE MAP: [Map ▼] Position ████████░░
LFO 1-3:   [Target▼] [Wave▼] Rate ██ Depth ██
```

### 粒子参数

| 参数 | 作用 | 范围 |
|------|------|------|
| Rate | 每秒触发粒子数 | 1–200 |
| Pitch | 基准音高 | 20–8000 Hz |
| Size | 粒子长度 | 3–500ms |
| Pitch Spread | 音高随机散布 | 0–±2 oct |
| Pan Spread | 声像随机散布 | 0–1 |
| Position | Timbre Map 读取位置 | 0(排列头)→1(排列尾) |
| Position Spread | 读取位置随机散布 | 0–1 |
| Feedback | 自采样混合 + 延迟 | 0–1 |

### Timbre Map 系统

2520 (exciter, body, modulator) combo 通过不同排列映射到一维。

| 排列 | Position 0→1 听感 |
|------|-------------------|
| Gradual | exciter 干净→复杂 (sine→snatch) |
| Dirty | exciter 反向 (snatch→sine) |
| Clean→Broken | body dry→pshift (短促→绵长破碎) |
| Strike→Sing | body 打击→吟唱 |
| Tuned→Noise | 谐波密度递减 |
| Tight→Vast | modulator 强度递增 (干→弥漫) |
| Full Jump | Morton Z-order (每格不同世界) |
| Random | shuffle(all) 加载时随机 |

每个排列是 `list[(e,b,m)]` 长度 2520。Position → `int(pos * 2519)` 索引。

### LFO 系统

3 个独立 LFO。每个配置：Target, Waveform (sine/tri/sqr/saw/S&H), Rate (0.1-20Hz), Depth (0-1)。

### 去除

LogisticMap, LorenzAttractor, RoesslerAttractor, ManifoldMapper, CellularAutomaton, LSystem, CouplingField, LongTermFeedback, compute_spectral_centroid, map_state_to_freq, map_tonic_spread_to_freq, map_dynamic_to_amp, quantize_to_scale, SCALES, run(), spatialize_*, attractor/spectrum UI。

### 保留

EXCITERS(12), BODIES(10), MODULATORS(7), VoicePool, SelfSampleBuffer, DelayNetwork, PresetManager, SAMPLE_RATE/BLOCK_SIZE。

---

## Task A1: 移除废弃代码

**Files:** `src/main.py`

删除：LogisticMap, LorenzAttractor, RoesslerAttractor, CHAOS_ENGINES, ManifoldMapper, map_state_to_freq, map_state_to_amp, CouplingField, LongTermFeedback, compute_spectral_centroid, SCALES, quantize_to_scale, map_tonic_spread_to_freq, map_dynamic_to_amp, CellularAutomaton, LSystem, run(), spatialize_*。

保留：SAMPLE_RATE, BLOCK_SIZE, EXCITERS dict, BODIES dict, MODULATORS dict, VoicePool, SelfSampleBuffer, DelayNetwork。

验证：`python -m py_compile src/main.py` 通过。

---

## Task A2: 新增 TimbreMap

**Files:** `src/main.py` (保留代码之后)

```python
class TimbreMap:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        all_combos = [(e,b,m) for e in range(12) for b in range(10) for m in range(7)]
        self.all_combos = all_combos
        self.maps = self._build_maps()
        self.current_map = 'Gradual'
        self._arrangement = self.maps[self.current_map]

    def _build_maps(self):
        maps = {}
        all_c = self.all_combos
        maps['Gradual'] = sorted(all_c, key=lambda c: (c[0], c[1], c[2]))
        maps['Dirty'] = sorted(all_c, key=lambda c: (-c[0], c[1], c[2]))
        maps['Clean→Broken'] = sorted(all_c, key=lambda c: (c[1], c[0], c[2]))
        body_order = {0:0,2:1,4:2,5:3, 1:4,3:5,6:6,7:7,8:8,9:9}
        maps['Strike→Sing'] = sorted(all_c, key=lambda c: (body_order[c[1]], c[0], c[2]))
        exc_tuned = {0:0,4:1,6:2,2:3,5:4,8:5,7:6,1:7,10:8,9:9,3:10,11:11}
        maps['Tuned→Noise'] = sorted(all_c, key=lambda c: (exc_tuned[c[0]], c[1], c[2]))
        maps['Tight→Vast'] = sorted(all_c, key=lambda c: (c[2], c[1], c[0]))
        # Morton Z-order
        def morton_key(c):
            def p1b2(n):
                n &= 0x3ff; n = (n|(n<<16))&0x30000ff; n = (n|(n<<8))&0x300f00f
                n = (n|(n<<4))&0x30c30c3; n = (n|(n<<2))&0x9249249; return n
            return p1b2(c[0])|(p1b2(c[1])<<1)|(p1b2(c[2])<<2)
        maps['Full Jump'] = sorted(all_c, key=morton_key)
        rng = np.random.RandomState(42)
        shuffled = all_c.copy(); rng.shuffle(shuffled)
        maps['Random'] = shuffled
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
```

验证：测试 get_combo(0)/get_combo(1) 在切换 map 后返回不同 combo。

---

## Task A3: 新增 LFO

**Files:** `src/main.py` (TimbreMap 之后)

```python
class LFO:
    WAVEFORMS = ['Sine','Triangle','Square','Saw','Random']
    def __init__(self, name="LFO1"):
        self.name = name; self.waveform='Sine'; self.rate=1.0
        self.depth=0.0; self.target='Pitch'
        self._phase=0.0; self._last_random=0.0; self._value=0.0

    def tick(self, dt):
        self._phase += self.rate * dt
        self._phase %= 1.0
        self._value = self._sample(self._phase)
        return self._value

    def _sample(self, phase):
        if self.waveform=='Sine': return float(np.sin(2*np.pi*phase))
        elif self.waveform=='Triangle': return float(4*abs(phase-0.5)-1)
        elif self.waveform=='Square': return 1.0 if phase<0.5 else -1.0
        elif self.waveform=='Saw': return float(2*phase-1)
        elif self.waveform=='Random':
            if phase < 0.01: self._last_random = float(np.random.uniform(-1,1))
            return self._last_random
        return 0.0

    @property
    def value(self):
        return self._value if self.depth>0.001 and self.rate>0.001 else 0.0
```

验证：tick(dt) 1000 次后 phase 在 [0,1]，各波形 value 在 [-1,1]。

---

## Task B1: 重写音频回调

**Files:** `src/ui/app.py`

新回调流程：
1. 读 params
2. tick 3 个 LFO
3. 计算 grains_this_block = int(rate * dt + accumulator)
4. 每个 grain: position→TimbreMap→combo, pitch×spread, size, pan_spread
5. exciter→body→ssb_mix→mod→trigger
6. render pool, ssb write, delay process

移除：chaos.step(), manifold.find_nearest(), material bias, generative layer, long-term feedback, spectral analysis, coupling field, sigma drift, silence injection, attractor queue。

验证：py_compile 通过。

---

## Task B2: 重写 UI

**Files:** `src/ui/app.py`

布局：
- PARTICLE 行: rate, pitch, size, feedback
- SPREAD 行: pitch_spread, pan_spread, position_spread  
- TIMBRE MAP 行: map selector combo + position slider
- LFO 行×3: target combo, wave combo, rate slider, depth slider

移除：attractor viz, spectrum plot, preset panel (后续再加)。

验证：启动 UI 无崩溃，所有控件可见可拖动。

---

## 验收

1. 无 chaos/Voronoi/generative/analysis 调用
2. Position 0→1 声随排列变化
3. 8 参数每个直接听觉反馈
4. 3 LFO 独立调制
5. 切换 Timbre Map 音色世界切换
6. Feedback 自采样+延迟有效
7. 5 分钟不崩溃
