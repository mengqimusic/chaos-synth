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
1. Chaos Engine 更新状态 → 输出吸引子坐标 (x, y, z) ∈ [0,1]³
2. Generative 层产出控制事件（可选叠加）
3. Manifold Mapper: Voronoi 流形映射 → 选择技术组合 (exciter+body+modulator+spatializer)
4. Unified Voice Pool: 128 统一语音池，每个语音 = 模块化微型合成器
5. Coupling Field: 语音间通过共享能量场耦合
6. Analysis Engine: FFT + 特征提取 → M 维特征向量
7. 特征向量反馈回 Chaos Engine / Manifold Mapper / Macros
8. 输出 buffer → 声卡
9. 下一帧...
```

### 2.2 架构分层

```
Control Layer (Chaos + Generative + External Input)
        │
        ▼
Manifold Mapper  ←── macros (Material, Density, Mutation, Coherence, Feedback, Energy)
  (Voronoi tessellation of [0,1]³)
        │
        ▼
  technique combo (exciter + body + modulator + spatializer)
        │
        ▼
Unified Voice Pool (128 voices, any combo)
        │           ↑
Coupling Field ─────┘  (shared energy buffer, ~1s)
        │
        ▼
Per-Voice Chaotic Evolution (logistic map per voice)
        │
        ▼           ↑
Analysis Engine ────┘  (feedback)
        │
        ▼
   Audio I/O → DAC / File
```

---

## 三、模块设计

### 3.0 核心概念：多拓扑粒子云 + 流形锚定

> **术语约定**: 在 Chaos Synth 中，**Voice (语音)** 和 **Particle (粒子)** 是同义词，指统一语音池中的一个微型合成器实例。
>
> **数量层级**: **池容量** (128 槽位，预分配数组) > **并发上限** (16–32，实际同时活跃，由 Density macro 限制) > **MVP 目标** (8，Phase 0 简化验证)。

#### 3.0.1 什么是多拓扑粒子

传统粒子合成中，每个粒子是从一条采样上截取的短窗口片段。所有粒子共享同一音色源泉。

**Chaos Synth 的多拓扑粒子**：每个粒子是一个**自主微型合成器实例**，由模块化组件组合而成。云团中的粒子是异质的——不同粒子用不同的技术组合，产生不同声学身份的声音，叠加后涌现出单一合成方式无法达到的复合音色。

```
传统粒子:   sample → N × [grain(windowed_sample)] → cloud
多拓扑粒子: attractor → manifold → N × [exciter+body+modulator+spatializer] → cloud
```

#### 3.0.2 锚点问题：流形替代采样

传统粒子合成中，采样提供三个功能：①音色一致性 ②时间方向性 ③声学结构。
没有采样意味着失去锚点——单纯用 seed → random 产生粒子参数，得到的只是随机噪声云，没有结构。

**解决方案：混沌吸引子 + Voronoi 流形 = 锚定轨道**

```
混沌吸引子 → (x, y, z) 坐标 ∈ [0,1]³
                    ↓
        Voronoi 流形映射器（把单位立方体切成 4-32 个细胞）
                    ↓
        最近质心 → 技术组合 (exciter + body + modulator + spatializer)
                    ↓
              具体声音参数 → Audio
```

核心洞察（参考 manifold-synth 设计）：**吸引子输出的 (a,b,c) 不是参数，是坐标。** 在 a 维度上走 0.48→0.52，可能跨过 Voronoi 细胞边界，从 FM spark→modal body 跳到 noise burst→comb body。仅 0.04 的坐标变化，声音完全质变。

这种不连续性是涌现多样性的核心——参数空间中的小步产生声音空间中的大跳跃。

**三层锚定体系：**

| 层级 | 锚点 | 作用 |
|------|------|------|
| **第一层** | 混沌吸引子轨道 + Voronoi 流形 | 轨道提供几何结构（Lorenz 双翼=两种音色区域），流形提供非线性映射（细胞边界=音色突变） |
| **第二层** | 频谱约束模板 | 设定演化目标（如 t=0 暗→t=30 亮→t=60 回归），给系统方向感 |
| **第三层** | 自采样（Phase 3+） | 系统录制输出到环形缓冲，喂回粒子引擎，活体吞噬 |

#### 3.0.3 模块化拓扑：技术组合库

不使用固定的完整拓扑（如 FM、Wavetable、Additive），而是用小模块组合：

**激发源（Exciter — 声音如何开始，12 种）：**

| # | 技术 | 特征 |
|---|------|------|
| 0 | Sine impulse | 纯正弦冲击，柔和瞬态 |
| 1 | Noise burst | 滤波白噪/粉噪/棕噪 |
| 2 | FM spark | 2-op FM，可配 carrier:modulator 比 |
| 3 | Granular micro | 5-30ms 内 2-8 个子 grain |
| 4 | Click | 锐利指数衰减瞬态 |
| 5 | Chirp | 线性/对数频率扫描 |
| 6 | Physical pluck | Karplus-Strong 波导管 |
| 7 | Wavetable scan | 短波表片段扫描 |
| 8 | Feedback ping | 短时反馈环自激振荡，临界参数附近产生不可预测瞬态 |
| 9 | Vowel burst | 2-3 共振峰噪声冲击，产生"人声感"粒子 |
| 10 | Ringmod spike | 极高频正弦相乘 (如 2000×2150 Hz)，3-10ms 金属瞬态 |
| 11 | Transient snatch | 从输出环形缓冲抓取峰值瞬态回放（Phase 3+ 活体吞噬） |

**共鸣体（Resonance Body — 声音如何持续/衰减，10 种）：**

| # | 技术 | 特征 |
|---|------|------|
| 0 | Dry | 简单指数衰减 |
| 1 | Modal bank | 2-6 个指数衰减正弦模态 |
| 2 | Comb | 反馈梳状滤波器（弦/镶边感） |
| 3 | Allpass | 扩散全通级联（混响感） |
| 4 | Nonlinear | 软限幅反馈环路 |
| 5 | Granular freeze | 短 buffer 循环 |
| 6 | Waveguide | 1D 波导（通用化 KS），模拟管/弦/棒/膜，长度可调 |
| 7 | Saturation chain | 3-5 级级联软限幅，驱动量递增 |
| 8 | Spectral blur | FFT→相位随机化→IFFT，模糊量可控 |
| 9 | Pitch-shifted FB | 延迟线 + 反馈路径中音高偏移 (shimmer)，产生 Shepard tone 感 |

**调制器（Modulator — 参数如何随时间演化，7 种）：**

| # | 技术 | 特征 |
|---|------|------|
| 0 | Static | 无调制 |
| 1 | Tremolo | 幅度调制 |
| 2 | Vibrato | 频率调制 |
| 3 | Morph | 语音中途渐变切换技术（crossfade 两个 body/exciter 输出） |
| 4 | Phase distortion | 非线性相位映射（Casio CZ 风格），混沌 r 值直接映射扭曲强度 |
| 5 | Envelope follower | 读取耦合场能量调制自身参数，产生"群体响应" |
| 6 | Cross-modulation | 粒子内部两参数互相调制，形成微混沌系统 |

**空间化器（Spatializer — 粒子在空间中的位置，3 种）：**

| # | 技术 | 特征 |
|---|------|------|
| 0 | Chaos pan | 声像由 attractor 坐标直接映射 |
| 1 | Haas cloud | 微小延迟偏移 (0-30ms) 替代声像，时间差营造空间感 |
| 2 | Distance decay | 声像 + 高频衰减 + 音量模拟"距离" |

**总组合空间：12 × 10 × 7 × 3 = 2520 种可能的粒子配置**

> **依赖链**: 下列技术依赖后续 Phase 的模块，在对应 Phase 之前标记为不可用：
> - Exciter #11 Transient snatch → 依赖自采样环形缓冲 (Phase 3+)
> - Modulator #3 Morph → 依赖并行渲染两个 body/exciter 输出 (Phase 2+)
> - Modulator #5 Envelope follower → 依赖耦合场 (Phase 1+)
>
> Phase 0 可用组合: 3 × 3 × 1 × 1 = 9 (指定技术)。Phase 2 可用完整 2520 组合。

**技术不可用时的回退策略**: 若 Voronoi 质心分配到不可用技术，按以下优先级回退：同一类中可用技术 → 同类 #0 → Dry/Static/Chaos pan。

#### 3.0.4 粒子调度与耦合

- **密度控制**: attractor 维度 → 0.5-50 grains/s
- **分布模式**: 均匀、Poisson、聚类、burst
- **生命周期**: 每个粒子 5ms-200ms，由 body 类型决定衰减特性
- **并发上限**: 16-32 个并发粒子
- **耦合场** (Phase 1+): ~1s 环形缓冲，每采样存储一个 RMS envelope 标量。粒子 `render()` 时读取当前缓冲位置的 envelope 值作为额外激励增益；渲染后将自身 `amp * feedback * 0.1` 加法沉积回当前位置。场值每采样衰减 `field *= (1 - coherence * 0.0005)`。时间邻近的粒子通过共享能量介质相互影响。Phase 0 不含耦合场
- **每语音混沌**: 每个粒子内部运行独立 logistic map，r 值由 Mutation macro 控制（3.5-3.95），产生生命周期内的有机微变化

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

### 3.3 Manifold Mapper（流形映射器）

将吸引子的 (x, y, z) 坐标映射到具体的技术组合。

**Voronoi 镶嵌：**

- 质心数量由 Density macro 映射：`count = 4 + Density × 28` → 范围 4-32
- 每个质心分配一个技术组合（exciter + body + modulator + spatializer）
- 触发时：找离坐标最近的质心 → 触发对应技术组合
- 质心位置由 seeded PRNG 计算，保证同一初始条件可复现

**Macros 对流形的控制：**

| Macro | 作用 |
|-------|------|
| **Material** | 整体偏移质心位置。低 Material → 质心聚集在打击/瞬态区域。高 Material → 聚集在持续/共振/质感区域。同坐标不同 Material = 完全不同音色 |
| **Density** | 质心数量。低 Density → 少数粗粒度细胞（声音类型少）。高 Density → 更多细粒度细胞（声音类型丰富） |
| **Mutation** | 质心位置抖动幅度。低 Mutation → 质心稳定可重复。高 Mutation → 每帧随机偏移 |
| **Coherence** | 质心稳定性。高 Coherence → 质心黏着在 attractor 附近。低 Coherence → 质心缓慢漂移 |
| **Feedback** | 质心历史依赖性。当前质心位置受前一帧质心影响 |
| **Energy** | 不直接影响质心，但影响技术组合中的激发幅度曲线 |

**非线性来源：** Voronoi 细胞边界天然产生不连续性——坐标的微小变化可导致技术组合的完全切换。这一机制是涌现多样性的核心。

### 3.4 Synthesis Engine（合成引擎）

统一语音池 (Unified Voice Pool)，128 个语音，每个语音由选中的技术组合驱动：

```python
class UnifiedVoice:
    active: bool
    phase: float          # 主振荡器相位
    age: int              # 触发后采样数
    duration: int         # 总生命周期
    amp: float            # 当前幅度
    pan: float            # 声像
    exciter: int          # 激发技术 0-11
    body: int            # 共鸣体技术 0-9
    modulator: int        # 调制技术 0-6
    spatializer: int      # 空间化技术 0-2
    state: int            # 状态机: attack/sustain/decay/mutation/done
    s: list[float]        # 技术状态数组
    chaos_x: float        # 每语音 logistic map 状态

    def trigger(self, combo, freq, amp, params): ...
    def render(self, buffer): ...
    def is_active(self) -> bool: ...
```

**语音窃取策略**：池满时驱逐最老的 active 语音（按触发时间排序的环形指针）。

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
[Logistic Map] → (x) coordinate ∈ [0,1]
       ↓
[1D Manifold Mapper: 4-point segmentation of [0,1]]
       ↓
[technique combo] → [Unified Voice Pool (8 concurrent voices)]
       ↓                   ↑
[Audio Out] → [FFT] → [Spectral Centroid feedback → modulate Logistic r]
```

> **维度策略**: Phase 0 使用 1D 流形（线段上 4 个分割点），Phase 1 升级到 2D/3D Voronoi。1D 坐标 (x) 直接对应声道 pan 和频率范围，无需固定 y=z=0。

**具体技术选择：**
- **Exciter (3 种)**: #0 Sine impulse, #1 Noise burst, #4 Click
- **Body (3 种)**: #0 Dry, #1 Modal bank, #2 Comb
- **Modulator (1 种)**: #0 Static
- **Spatializer (1 种)**: #0 Chaos pan（坐标直接映射声像）
- **组合**: 3×3×1×1 = 9

**冷启动**: 系统启动时，若连续 500ms 内 RMS < 阈值，自动注入一次噪声底噪（-40dB），确保反馈闭环不会陷入永久静默。

- Logistic map (1D) 驱动坐标，r 的初始值 = 3.7（混沌边缘）
- 4 质心 1D 流形分割 [0,1] 区间
- 8 个并发语音，共享池容量 128
- 实时输出，验证闭环稳定
- **~400 行 Python**

### Phase 1: 混沌多元 + 模块扩展

- Lorenz / Rössler 吸引子
- 流形扩展到 16 质心
- 扩展 exciter 到 8 种，body 到 6 种，modulator 到 4 种
- 多特征反馈（centroid→频率, flux→振幅, ZCR→FM depth）
- 耦合场基础实现
- TOML 配置文件系统

### Phase 2: 生成式控制

- 元胞自动机驱动节奏（每迭代取 8 位作为触发 pattern）
- L-System 驱动音高序列（axiom + 2-3 条 production rule，每次推导输出一个 MIDI note）
- 多声部架构 (3-8 独立旋律/节奏声部，每声部管理自己的语音子池，声部间可耦合)
- 全模块库 (12×10×7×3 = 2520 组合)
- 基础 GUI：参数面板 + 实时频谱

### Phase 3: 颗粒与深度反馈

- 自采样闭环（Transient snatch exciter）
- 多重反馈路径 (short/long-term)
- 延迟网络 (No-input mixer 风格)
- 耦合场的完整非线性实现
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
| 2520 组合参数空间过大 | 流形质心数量由 Density 控制，low=4 组合 |
| 系统冷启动陷入静默 | 静默检测 (500ms RMS<阈值) → 自动注入 -40dB 噪声底噪 |

---

## 八、设计参考

- manifold-synth 设计 (`chaos-llm-synth/docs/superpowers/specs/2026-05-13-manifold-synth-design.md`)
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
│   ├── engine/
│   │   ├── audio_io.py       # sounddevice 封装
│   │   ├── voice_pool.py     # 128 统一语音池
│   │   ├── coupling_field.py # 耦合场: 1s 能量缓冲
│   │   ├── manifold.py       # Voronoi 流形映射器
│   │   ├── analysis.py       # FFT + 特征提取
│   │   └── feedback.py       # 反馈循环调度
│   ├── techniques/
│   │   ├── exciter.py        # 激发源库 (12)
│   │   ├── body.py           # 共鸣体库 (10)
│   │   ├── modulator.py      # 调制器库 (7)
│   │   └── spatializer.py    # 空间化器库 (3)
│   ├── chaos/
│   │   ├── logistic.py
│   │   ├── lorenz.py
│   │   ├── roessler.py
│   │   └── base.py
│   ├── generative/
│   │   ├── cellular.py
│   │   └── lsystem.py
│   ├── mapping/
│   │   └── params.py         # 参数映射策略
│   └── ui/
│       └── spectrum.py
├── tests/
│   ├── test_chaos.py
│   ├── test_manifold.py
│   └── test_voice_pool.py
└── docs/
    └── chaos-synth-plan.md
```

---

## 十、下一步

1. ✅ 策划方案完成（含流形映射 + 模块化拓扑 2520 组合）
2. 确认技术选型 — Python 原型 OK？
3. 确认 MVP 范围 — Phase 0 包含 4-cell Voronoi + 9 组合 + 8 并发语音？
4. 开始实现 — 按 writing-plans skill 拆成 bite-size tasks

---

> **一句话**: 这不是乐器，这是一个会自己演奏的音乐生态系统——现在有了 2520 种可能的声学形态。
