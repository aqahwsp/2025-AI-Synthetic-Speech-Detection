# 用于2025 阿里天池全球 AI 攻防挑战赛  
## 泛终端智能语音交互认证 - 检测赛（伪造音频检测）

本项目面向 **AIGC 语音伪造检测** 场景，目标是在复杂真实环境下识别 `Spoof`（伪造）与 `Bonafide`（真实）语音。  
仓库包含完整的数据检查、自动训练、模型变体实验与评估流程，适合作为比赛基线与工程原型。

---

## 项目亮点

- **端到端检测框架**：`CNN + BiGRU + 多头查询注意力池化`
- **弱定位增强**：引入帧级 `Top-K MIL` 约束，强调局部异常片段
- **可解释性**：输出注意力权重，支持分析模型关注时段
- **工程化训练**：自动发现数据目录、自动 CSV 对齐、自动 7:3 划分
- **鲁棒增强版本**：双视角训练 + 噪声增强（`aigc_detector0.py / 1.py`）

---

## 方法概览

### 1) 特征与主干网络
- 输入：16kHz 单声道语音
- 特征：`64-dim Log-Mel` 频谱
- 主干：
  1. CNN 频谱前端（压缩频率维，不降采样时间维）
  2. BiGRU 建模时序依赖
  3. Multi-Head Query Attention 在时间维做聚焦池化
  4. Clip-level 分类头输出伪造概率

### 2) 损失设计
- 片段级二分类损失（BCE）
- 帧级 `Top-K MIL` 损失（强调局部伪造痕迹）
- 注意力熵正则（鼓励更聚焦的注意力分布）

> 标签定义（代码中固定）：`Bonafide=0`，`Spoof=1`（Spoof 为正类）

### 3) 增强版（0/1）
- 双视角训练：整段视角 + 短切片视角
- 噪声增强：pre-emphasis / high-pass / 随机 SNR 加噪
- 更灵活的网络参数（如 CNN 深度、GRU 层数等）

---

## 仓库结构

```text
.
├── aigc_detector.py        # 基线模型（CNN+BiGRU+注意力池化+MIL）
├── aigc_detector0.py       # 增强版模型 0（双视角+噪声增强）
├── aigc_detector1.py       # 增强版模型 1（进一步可配）
├── auto_train.py           # 基线自动训练脚本（推荐 global 模式）
├── auto_train0.py          # 对应 detector0 的自动训练
├── auto_train1.py          # 对应 detector1 的自动训练
├── check_csv_formats.py    # CSV 规范检查/可选自动规范化
├── len.py                  # 统计 WAV 长度（平均/最短）
└── tmp.py                  # 简单 CSV 第三列标签计数工具
