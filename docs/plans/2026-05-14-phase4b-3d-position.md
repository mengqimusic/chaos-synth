# Phase 4b — 3D Timbre Position + 独立 Position LFO

> **For Hermes:** 逐任务 subagent 实现，每任务审核后提交。

**Goal:** 去掉 TimbreMap，Position 改为 3 维（Pos E / Pos B / Pos M），各轴独立 0-1 映射到 exciter/body/modulator。Pitch 改为 MIDI 半音，Size 上限提升到 2s。专用 3 个 LFO 分别调制 3 个 position 轴 + 3 个通用 LFO。

**Architecture:** main.py 删除 TimbreMap 类。app.py 重写 UI 和回调。

---

## 参数设计

### PARTICLE
| 参数 | 范围 | 默认 | 说明 |
|------|------|------|------|
| Rate | 1–200 | 20 | grains/s |
| Pitch | 21–108 (MIDI) | 60 (C4) | 半音刻度 |
| Size | 0.001–2.0 | 0.1 | 秒 |
| Feedback | 0–1 | 0.3 | 自采样混合 |

### POSITION（3D 音色空间）
| 参数 | 范围 | 默认 | 映射到 |
|------|------|------|--------|
| Pos E | 0–1 | 0.5 | exciter 0-11 |
| Pos B | 0–1 | 0.5 | body 0-9 |
| Pos M | 0–1 | 0.5 | modulator 0-6 |
| Pos Spread | 0–1 | 0.1 | 3 轴共用随机散布 |

```python
eid = int(np.clip(pos_e + np.random.uniform(-spread, spread), 0, 0.999) * 12)
bid = int(np.clip(pos_b + np.random.uniform(-spread, spread), 0, 0.999) * 10)
mid = int(np.clip(pos_m + np.random.uniform(-spread, spread), 0, 0.999) * 7)
```

### SPREAD
| 参数 | 范围 | 默认 | 说明 |
|------|------|------|------|
| Pitch Spread | 0–24 半音 | 2 | ±半音随机散布 |
| Pan Spread | 0–1 | 0.5 | 声像散布 |

```python
pfreq = midi_to_hz(pitch + random.uniform(-spread, spread))
```

midi_to_hz: `440 * 2**((note - 69) / 12)`

### LFO 布局（6 个）

```
POS LFO E:  [Wave▼] Rate Depth ──→ Pos E
POS LFO B:  [Wave▼] Rate Depth ──→ Pos B
POS LFO M:  [Wave▼] Rate Depth ──→ Pos M
LFO 1:      [Target▼] [Wave▼] Rate Depth
LFO 2:      [Target▼] [Wave▼] Rate Depth
LFO 3:      [Target▼] [Wave▼] Rate Depth
```

前 3 个固定调制 position 轴，无需 target 选择。后 3 个可选 target: Pitch/Size/Rate/Pan/Pitch Spread/Pos Spread。

---

## Task A1: 删除 TimbreMap

**Files:** `src/main.py`

删除 TimbreMap 类（约 537-584 行）。LFO 类保留不动。

验证：`py_compile` 通过。提交。

---

## Task B1: 重写 app.py

**Files:** `src/ui/app.py`

完全重写：

1. **import**: 删除 TimbreMap，保留 LFO,VoicePool,SelfSampleBuffer,DelayNetwork,EXCITERS,BODIES
2. **_params**: 按新参数表重写
3. **引擎实例**: 删除 `_timbre_map`，新增 6 个 LFO（`_pos_lfos[3]` + `_lfos[3]`）
4. **_audio_callback**: 
   - midi_to_hz 转换
   - position 3D 映射 combo
   - 前 3 个 LFO 固定调 position 轴
   - 后 3 个 LFO 调任意参数
   - grain 触发逻辑保持不变
5. **_build_ui**: 按新布局重写

UI 布局：
```
PARTICLE:  Rate | Pitch(21-108) | Size(0.001-2) | Feedback
POSITION:  Pos E | Pos B | Pos M | Pos Spread
SPREAD:    Pitch Spread(0-24sem) | Pan Spread
POS LFO:   E: [Wave] Rate Depth | B: [Wave] Rate Depth | M: [Wave] Rate Depth
LFO 1-3:   [Target] [Wave] Rate Depth (×3)
```

Pitch slider 用 int 格式显示 MIDI note number。

验证：`py_compile` 通过，提交。
