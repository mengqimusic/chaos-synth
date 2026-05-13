# Chaos Synth — 算法反馈合成器 策划方案

> **核心命题**: 使用算法实时分析音频输出，将分析结果反馈调制合成参数，形成自激振荡的混沌乐音系统。不使用传统键盘界面，以生成式算法驱动。

---

## 一、核心理念

### 1.1 什么是"算法反馈合成"

传统合成器：演奏者输入 → 合成参数 → 声音输出（单向）

**Chaos Synth**：合成器输出 → 实时频谱分析 → 特征提取 → 非线性映射 → 合成参数调制 → 新输出（闭环反馈）

声音不再是"被弹出来的"，而是在系统中**自发生长**的。输入端不是键盘，而是：
- 初始条件（种子参数）
- 混沌系统的演化轨迹
- 生成式算法（元胞自动机、L-system、随机游走）
- 外部环境输入（麦克风、传感器、网络数据流）

### 1.2 乐音 × 混沌

| 维度 | 乐音（Cosmos） | 混沌（Chaos） |
|------|----------------|---------------|
| 音高 | 调性中心、音阶约束 | 失谐、微分音、滑音 |
| 节奏 | 格律、拍号 | 非整数时值、概率节奏 |
| 音色 | 谐波系列 | 噪声、非谐波、粒子 |
| 结构 | 主题发展 | 分形自相似、涌现 |

**关键原则**：不是随机，而是**确定性混沌**——系统是确定性的（无随机数），但因非线性反馈而不可预测。这会产生"有组织的复杂"：介于纯噪声和纯音乐之间的第三领域。

---

## 二、系统架构

### 2.1 核心循环（每帧 ~5.8ms @ 44.1kHz, 256 samples）

```
1. Chaos Engine 更新状态 → 输出 N 维向量
2. Generative 层产出控制事件（可选）
3. Parameter Mapper 映射到 0..1 合成参数空间
4. Synthesis Engine 生成 256-sample buffer
5. Analysis Engine FFT + 特征提取 → M 维特征向量
6. 特征向量反馈回 Chaos Engine / Parameter Mapper
7. 输出 buffer → 声卡
8. 下一帧...
```

### 2.2 架构分层

```
Control Layer (Chaos + Generative + External Input)
        ↓
Parameter Mapper (Normalize, Quantize, Scale/Curve)
        ↓           ↑
Synthesis Engine    │  (feedback)
        ↓           │
Analysis Engine ────┘
        ↓
   Audio I/O → DAC / File
```

---

## 三、模块设计



### 0. 核心概念：多拓扑粒子云（Poly-Topological Particle Cloud）

#### 0.1 什么是多拓扑粒子

传统粒子合成中，每个粒子是从一条采样上截取的短窗口片段。所有粒子共享同一音色源泉。

**Chaos Synth 的多拓扑粒子**：每个粒子是一个**自主微型合成器实例**，拥有独立的合成拓扑（FM、wavetable、additive、physical model、noise 等），从零生成自己的波形。云团中的粒子是异质的——不同粒子用不同的合成方法，产生不同声学身份的声音，叠加后涌现出单一合成方式无法达到的复合音色。

```
传统粒子:   sample → N × [grain(windowed_sample)] → cloud
多拓扑粒子: attractor → N × [micro_synth(type, params)] → cloud
```

#### 0.2 锚点问题：没有采样怎么办

传统粒子合成中，采样提供三个功能：①音色一致性 ②时间方向性 ③声学结构。
没有采样意味着失去锚点——单纯用 seed → random 产生粒子参数，得到的只是随机噪声云，没有结构。

**解决方案：三层锚定体系**

**第一层 — 混沌吸引子轨道（主锚）**

吸引子在相空间中的轨迹替代了采样的角色。它不是录音，而是一条在合成参数空间中有结构、有边界、有几何形态的轨道：

```
Lorenz 双翼结构 = 两种"音色区域"
轨道在同一个翼里盘旋 = 音色统一的段落
轨道跳跃到另一个翼 = 音色突变 / 转调
```

每一帧，吸引子的 N 维状态直接决定：
- 用哪个合成拓扑（粒度选择）
- 该粒子的频率、振幅、包络（参数填充）
- 触发时机和密度（调度决策）

同一个初始条件 → 同一条确定性轨道 → 同一段可复现的"音乐"。

**第二层 — 频谱约束模板（副锚）**

不依赖采样，但给系统设定一个频谱演化目标：

```
模板:
  t=0s:  低频主导，窄带
  t=30s: 全频段激活，宽带
  t=60s: 回归窄带低频

反馈: 当前云团频谱 vs 目标频谱 → 误差 → 调制吸引子参数
```

模板没有具体波形，但给了系统**方向感**。系统朝着频谱目标演化，中间路径是混沌的。

**第三层 — 自采样 / 活体吞噬（Phase 3+）**

系统实时录制自己的输出到环形缓冲区，作为"采样"喂回粒子引擎——系统吞噬自己的产出。

#### 0.3 粒子拓扑库（初期 6 种）

| 拓扑 | 复杂度 | 特征音色 |
|------|--------|----------|
| **FM-2op** | 低 | 金属、钟声、谐波/非谐波频谱 |
| **Wavetable** | 中 | 平滑扫描、Pad、可塑性强 |
| **Subtractive** | 低 | 温暖、经典合成器、噪声+滤波 |
| **Additive-8** | 中 | 纯净、管风琴、精确谐波控制 |
| **KS-Pluck** | 低 | 拨弦、弹拨、短促打击 |
| **Noise-Filtered** | 极低 | 风、海、呼吸、质感噪声 |

所有拓扑实现统一接口：

```python
class MicroSynth:
    def trigger(self, freq, amp, params): ...  # 触发一个粒子
    def render(self, buffer): ...               # 填充 audio buffer
    def is_active(self) -> bool: ...            # 粒子是否仍在发声
```

#### 0.4 粒子调度

- **密度控制**: attractor 的一个维度映射到 0.5-50 grains/s
- **分布模式**: 均匀、Poisson、聚类、burst
- **生命周期**: 每个粒子 5ms-200ms，指数衰减包络
- **并发上限**: 初期 16-32 个并发粒子，CPU 预算可控
- **全局后处理**: 所有粒子汇总后通过共享效果链（reverb、filter、compressor）确保云团的空间一致


### 3.1 Chaos Engine（混沌引擎）

| 系统 | 维度 | 特征 | 音乐用途 |
|------|------|------|----------|
| Logistic Map | 1D | 倍周期分岔 → 混沌 | 振幅、滤波截止 |
| Lorenz Attractor | 3D | 双翼结构、蝴蝶效应 | 立体声像、3 参数联动 |
| Rössler | 3D | 螺旋混沌 | 频率调制、滤波器扫频 |
| Chua's Circuit | 3D | 双涡卷 | 波形折叠、非线性失真 |
| Hénon Map | 2D | 奇异吸引子 | 节奏触发、门限 |
| Ikeda Map | 2D | 光学混沌 | 相位调制、延迟时间 |

每个系统暴露内部参数（σ, ρ, β 等），可被反馈调制。

### 3.2 Generative Control（生成式控制）

| 算法 | 用途 | 控制对象 |
|------|------|----------|
| 1D 元胞自动机 (Rule 30/110) | 复杂节奏序列 | 触发包络、音符开关 |
| 2D CA (Game of Life) | 空间化音色 | 多声部激活/静音 |
| L-System | 旋律轮廓生成 | 音高序列 |
| 马尔可夫链 | 学习+生成模式 | 和声进行、参数序列 |
| Lévy Flight | 极端事件触发 | 突发音效、段落切换 |
| Perlin/Simplex 噪声 | 连续平滑调制 | 滤波、混响、声像 |

### 3.3 Parameter Mapper（参数映射）

```
映射策略:
- Linear:      output = input * range + offset
- Exponential: output = range * (input^curve)
- Quantize:    output = round_to_scale(input, scale_degrees)
- Bipolar:     output = (input - 0.5) * 2 * range
- Hysteresis:  仅在输入变化 > threshold 时更新
- Smooth:      低通滤波后的参数 (slew rate limit)
```

### 3.4 Synthesis Engine（合成引擎）

| 算法 | 说明 | 优先级 |
|------|------|--------|
| **Wavetable** | 动态波形表 + 插值 | ⭐⭐⭐ |
| **FM Synthesis** | 2/4/6-op，频率比非整数 | ⭐⭐⭐ |
| **Additive** | 8-32 个独立调制正弦谐波 | ⭐⭐ |
| **Granular** | 实时录制 → 粒子化 → 反馈重注 | ⭐⭐⭐ |
| **Waveshaping** | Chebyshev / 查表失真 | ⭐⭐ |
| **Subtractive** | 噪声 → 共振滤波 → 反馈 | ⭐⭐ |
| **Karplus-Strong** | 物理建模拨弦 → 反馈驱动 | ⭐ |

### 3.5 Analysis Engine（分析引擎）

| 特征 | 计算 | 映射目标 |
|------|------|----------|
| RMS Energy | sqrt(mean(x²)) | 振幅、压缩 |
| Spectral Centroid | Σ(f·|X|) / Σ|X| | 滤波截止、亮度 |
| Spectral Flux | Σ(|X_t| - |X_{t-1}|)² | 瞬态检测、包络触发 |
| Zero-Crossing Rate | count(x_n·x_{n-1} < 0) / N | 噪声度、复杂度 |
| Pitch (YIN) | 自相关基频估计 | 音高量化、和声约束 |
| Crest Factor | peak / RMS | 动态范围、压缩比 |
| Spectral Spread | std(f, weighted by |X|) | 带宽、Q值 |

---

## 四、技术栈

### Python 原型（推荐）

| 层 | 库 | 理由 |
|----|-----|------|
| Audio I/O | `sounddevice` | 跨平台、低延迟回调 |
| DSP | `numpy` + `scipy.signal` | 向量化 |
| FFT | `numpy.fft` | 标准库 |
| 可视化 | `pyqtgraph` | 实时频谱 |
| MIDI | `python-rtmidi` | 外部控制器 |

### 长期路径

Phase 1: Python 原型 (1-2 周) → Phase 2: numba 加速 (1 周) → Phase 3: Rust 重写核心 (2-4 周) → Phase 4: Daisy Seed 移植

---

## 五、迭代路线

### Phase 0: MVP — 跑通反馈循环 ⭐

```
[oscillator] → [audio out]
     ↑              ↓
[param map] ← [FFT analysis]
     ↑
[Logistic Map]
```

- 单个 sine oscillator
- Logistic map 控制频率
- Spectral centroid 反馈调制 logistic map 的 r 参数
- 实时输出，验证闭环稳定
- **~200 行 Python，单文件**

### Phase 1: 混沌多元

- Lorenz / Rössler 吸引子
- 多振荡器（FM、wavetable）
- 多特征反馈（centroid→频率, flux→振幅, ZCR→FM depth）
- TOML 配置文件系统

### Phase 2: 生成式控制

- 元胞自动机驱动节奏
- L-System 驱动音高序列
- 多声部 (3-8 voices)
- 基础 GUI：参数面板 + 实时频谱

### Phase 3: 颗粒与深度反馈

- 实时 granulator
- 多重反馈路径 (short/long-term)
- 延迟网络 (No-input mixer 风格)
- 预设管理 + morphing

### Phase 4: 硬件移植

- Rust 重写核心 DSP
- Bela / Daisy Seed 平台测试
- CV/Gate 输出 (Eurorack)

---

## 六、非键盘界面方案

| 方案 | 硬件 | 描述 |
|------|------|------|
| 触摸参数面板 | 触摸屏/iPad | 多点触控调制混沌参数 |
| MIDI 旋钮/推子 | MIDI 控制器 | 8-16 旋钮映射 chaos params |
| 麦克风驱动 | 麦克风 | 环境声包络/频谱驱动参数 |
| OSC 网络控制 | 手机/平板 | Wi-Fi 发送 OSC 消息 |
| 纯算法自动 | 无 | 无需交互，系统自我演化 |
| 手势/摄像头 | Webcam | MediaPipe 手势映射参数 |

MVP 阶段：命令行 + 配置文件 + 频谱显示。MIDI 控制尽早加入。

---

## 七、风险与对策

| 风险 | 对策 |
|------|------|
| 反馈不稳定/啸叫 | 限幅器 + soft clip；参数空间约束 |
| 实时性能瓶颈 | 增大 buffer；numba JIT；C 扩展 |
| 混沌不可控(永远噪声) | "混沌度" meta-parameter 平滑过渡 |
| Python GIL | sounddevice callback 在独立 C 线程 |
| 硬件移植推倒重来 | 核心算法纯函数，I/O 抽象层 |

---

## 八、灵感来源

- Chowning, FM Synthesis (1973)
- Roads, "Microsound" (2001)
- Agostino Di Scipio — audible ecosystemics
- Toshimaru Nakamura — no-input mixing board
- Brian Eno — Generative Music
- SuperCollider, Pure Data, Max/MSP
- Mutable Instruments, DaisySP

---

## 九、项目结构

```
chaos-synth/
├── README.md
├── pyproject.toml
├── config/presets/
├── src/
│   ├── main.py
│   ├── engine/        (audio_io, synthesis, analysis, feedback)
│   ├── chaos/          (logistic, lorenz, base)
│   ├── generative/     (cellular, lsystem)
│   ├── mapping/        (params)
│   └── ui/             (spectrum)
├── tests/
│   ├── test_chaos.py
│   ├── test_synthesis.py
│   └── test_analysis.py
└── docs/
```

---

## 十、下一步

1. ✅ 策划方案完成
2. 确认技术选型 — Python 原型 OK？
3. 确认 MVP 范围 — Phase 0 单文件反馈循环？
4. 开始实现 — 按 writing-plans skill 拆成 bite-size tasks

---

> **一句话**: 这不是乐器，这是一个会自己演奏的音乐生态系统。
