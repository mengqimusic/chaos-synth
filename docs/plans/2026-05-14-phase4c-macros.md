# Phase 4c — OP-1 Style Macros for E/B/M Modules

> **For Hermes:** 逐任务 subagent 实现，审核后提交。

**Goal:** 为 Exciter/Body/Modulator 各添加 2 个角色化 Macro 旋钮，直接控制模块内部参数，无需理解内部实现即可塑造音色。

**Architecture:** 修改 main.py 中各函数签名增加 macro 参数，VoicePool.trigger() 透传，app.py 添加 6 个 macro 滑块。

---

## Macro 定义

### Exciter Macros

| Macro | 0 | 1 | 通用语义 |
|-------|---|---|---------|
| **Bite** | 软起音 | 锋利瞬态 | attack sharpness |
| **Color** | 暗/纯 | 亮/丰富 | harmonic warmth |

| Exciter | Bite 控制 | Color 控制 |
|---------|----------|-----------|
| 0 sine | attack ratio: 1/4→1/16 | harmonics: 1→5 sines |
| 1 noise | envelope: hann→rectangle | noise color: brown→white→blue |
| 2 FM | mod index: 1→8 | mod ratio: 1:2→1:3.7 |
| 3 granular | sub-grains: 2→8 | pitch variance: 0→0.5 |
| 4 click | width: 4→1 sample | resonance tail: 0→0.5 |
| 5 chirp | sweep ratio: ×2→×8 | curve: linear→exponential |
| 6 pluck | excitation pos: 0.5→0.1 | damping: 0.99→0.5 |
| 7 wavetable | harmonic count: 2→8 | waveform morph: sine→saw |
| 8 fb_ping | feedback: 0.7→0.99 | delay: 20ms→5ms |
| 9 vowel | bandwidth(Q): 200→50Hz | formant shift: 0.8→1.2 |
| 10 ringmod | duration: 10→3ms | carrier ratio: 0.5→4.0 |
| 11 snatch | position: avg→peak | blend: dry→wet |

### Body Macros

| Macro | 0 | 1 | 通用语义 |
|-------|---|---|---------|
| **Ring** | 干短 | 长鸣 | decay/resonance time |
| **Body** | 小腔体 | 大腔体 | cavity size/pitch |

| Body | Ring 控制 | Body 控制 |
|------|----------|----------|
| 0 dry | tau: 10ms→500ms | — (no cavity) |
| 1 modal | tau scale: 0.2→3.0 | freq ratio: 1.0→3.0 |
| 2 comb | feedback: 0.3→0.95 | delay length: ×2→×0.5 |
| 3 allpass | stage count: 1→8 | delay length: ×2→×0.5 |
| 4 nonlinear | feedback: 0→0.9 | drive: 1.0→6.0 |
| 5 freeze | loop iterations: 1→8 | loop length: 16→128 |
| 6 waveguide | damping: 0.99→0.5 | delay: ×1.5→×0.5 |
| 7 saturation | stage count: 1→6 | drive: 1.0→6.0 |
| 8 blur | mix: 0→1.0 | window: 512→64 |
| 9 pshift_fb | feedback: 1→8 iterations | shift: 1.001→1.05 |

### Modulator Macros

| Macro | 0 | 1 | 通用语义 |
|-------|---|---|---------|
| **Warp** | 干净 | 深度调制 | modulation intensity |
| **Grit** | 纯净 | 破坏 | distortion/crush amount |

| Modulator | Warp 控制 | Grit 控制 |
|-----------|----------|----------|
| 0 static | — | — |
| 1 tremolo | depth: 0→1.0 | clip: 0→0.5 |
| 2 vibrato | depth: 0→0.02 | jitter: 0→0.01 |
| 3 phase_dist | distortion: 0→1.0 | feedback: 0→0.5 |
| 4 ringmod | depth: 0→1.0 | freq: 50→2000Hz |
| 5 bitcrush | — | bits: 16→2 |
| 6 stereo_width | width: 0→1.0 | — |

---

## Task 1: 修改 Exciter 函数签名（加 bite, color）

**Files:** `src/main.py`

每个 exciter 函数在现有 `freq, sr` 后添加 `bite=0.5, color=0.5` 默认参数。

实现策略：
- 大部分 exciter 中，bite 和 color 默认值 0.5 时行为不变（映射到原始硬编码值的中点）
- bite 和 color 改变时，线性插值到新范围

以 exciter_fm_spark 为例：
```python
def exciter_fm_spark(freq, sr=SAMPLE_RATE, bite=0.5, color=0.5):
    mod_idx = 1.0 + bite * 7.0       # 1→8 (orig:3→中点在4.5)
    mod_ratio = 2.0 + color * 1.7    # 2→3.7 (orig:2→中点)
    ...  # 其余不变，但用 mod_idx 替代硬编码的 3
```

以 exciter_noise_burst 为例：
```python
def exciter_noise_burst(freq, sr=SAMPLE_RATE, bite=0.5, color=0.5):
    # bite: hann→rectangle (0=hann, 1=rectangular)
    window = np.hanning(512) * (1-bite) + np.ones(512) * bite
    # color: brown→white→blue
    noise = np.random.randn(512)
    # 0=brown(lowpass), 0.5=white, 1=blue(highpass)
    if color < 0.5:
        # simple 1-pole lowpass
        ...
    else:
        # simple 1-pole highpass
        ...
```

验证：每个 exciter 修改后 `py_compile` 通过。不 commit（等全部改完一起提交）。

---

## Task 2: 修改 Body 函数签名（加 ring, body）

**Files:** `src/main.py`

同 Task 1 模式。每个 body 函数添加 `ring=0.5, body_size=0.5`。

验证：py_compile 通过。不 commit。

---

## Task 3: 修改 Modulator 函数签名（加 warp, grit）

**Files:** `src/main.py`

每个 modulator 添加 `warp=0.5, grit=0.5`。

验证：py_compile 通过。不 commit。

---

## Task 4: 更新 VoicePool.trigger() 透传 macros

**Files:** `src/main.py`

`trigger()` 方法添加 `bite, color, ring, body_size, warp, grit` 参数，透传给 exciter/body/modulator 调用。

```python
def trigger(self, exciter_id, body_id, modulator_id, freq, amp, chaos_x,
            bite=0.5, color=0.5, ring=0.5, body_size=0.5, warp=0.5, grit=0.5):
    efn = EXCITERS.get(exciter_id, exciter_sine_impulse)
    excitation = efn(freq, SAMPLE_RATE, bite=bite, color=color)
    bfn = BODIES.get(body_id, body_dry)
    grain = bfn(excitation, freq, SAMPLE_RATE, ring=ring, body_size=body_size)
    # modulator with warp, grit
    ...
```

验证：py_compile 通过。不 commit。

---

## Task 5: 更新 app.py UI 和回调

**Files:** `src/ui/app.py`

1. `_params` 添加 6 个 macro 默认值（均 0.5）
2. UI 中添加 MACRO 行，6 个滑块（Bite/Color/Ring/Body/Warp/Grit）
3. 回调中读取 macros 并传给 `_pool.trigger()`

UI 布局：
```
MACRO:  Bite | Color | Ring | Body | Warp | Grit
```

标签简洁，不放详细说明（OP-1 哲学）。

验证：py_compile + 回调测试通过。提交全部修改。
