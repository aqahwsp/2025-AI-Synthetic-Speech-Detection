# -*- coding: utf-8 -*-
"""
aigc_detector.py

AIGC 语音伪造检测：CNN+BiGRU + 多头注意力池化（关注局部异常）
- 训练：train(...)
- 推理：infer_to_csv(...)
- 评估：evaluate_predictions(...)

数据集格式：
每个压缩包解压后含若干 wav 和一个 csv，csv 列：utt,path,label（label ∈ {Spoof, Bonafide}）
root_dir/part_xxx/*.wav
root_dir/part_xxx/labels_xxx.csv

本实现聚焦“可用性与可解释性”，适合作为比赛/生产基线。
"""
import os
import csv
import math
import json
import random
import warnings
from typing import List, Tuple, Optional, Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# 尽量同时兼容 torchaudio 与 librosa，缺哪个就用另一个
try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False
    warnings.warn("torchaudio 未安装，使用 librosa 读取音频。建议安装 torchaudio 获得更快的加载。")
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False
    raise ImportError("至少需要安装 librosa 或 torchaudio 之一。")

################################################################################
# 工具函数
################################################################################

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """读取 wav -> 单声道 float32, 归一化到 [-1,1]，重采样到 target_sr"""
    if _HAS_TORCHAUDIO:
        wav, sr = torchaudio.load(path)  # [C, T]
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav.squeeze(0).numpy().astype(np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        return wav
    else:
        wav, sr = librosa.load(path, sr=None, mono=True)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        wav = wav.astype(np.float32)
        wav = np.clip(wav, -1.0, 1.0)
        return wav

def wav_to_logmel(
    wav: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,          # 25ms
    hop_length: int = 160,     # 10ms
    n_mels: int = 64,
    fmin: int = 20,
    fmax: Optional[int] = 7600,
    eps: float = 1e-10,
) -> np.ndarray:
    """转为对数 Mel 频谱，返回形状 [n_mels, T]"""
    if _HAS_TORCHAUDIO:
        wav_t = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, f_min=fmin, f_max=fmax, n_mels=n_mels,
            power=2.0, center=True, pad_mode='reflect'
        )
        mel = mel_tf(wav_t)  # [1, n_mels, T]
        mel = torch.log(mel + eps).squeeze(0).numpy()
        return mel
    else:
        S = librosa.feature.melspectrogram(
            y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0,
            center=True
        )
        logmel = np.log(S + eps).astype(np.float32)
        return logmel

def pad_stack(feats: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 [C,T] 序列 pad 到同一长度，返回 (batch, C, T) 与有效长度 [B]"""
    lengths = torch.tensor([x.shape[-1] for x in feats], dtype=torch.long)
    max_len = int(lengths.max().item())
    C = feats[0].shape[0]
    batch = torch.full((len(feats), C, max_len), pad_value, dtype=torch.float32)
    for i, x in enumerate(feats):
        t = x.shape[-1]
        batch[i, :, :t] = x
    return batch, lengths

def mask_by_length(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, ...], lengths: [B]
    返回布尔 mask: True 为有效位置
    """
    B, T = x.shape[0], x.shape[1]
    idxs = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    mask = idxs < lengths.unsqueeze(1)
    return mask  # [B, T]

################################################################################
# 数据集
################################################################################

LABEL2ID = {"Bonafide": 0, "Spoof": 1}
ID2LABEL = {0: "Bonafide", 1: "Spoof"}

class AIGCDataset(Dataset):
    def __init__(
        self,
        csv_paths: List[str],
        roots: List[str],
        sr: int = 16000,
        # 训练：按比例随机裁剪（例如 60%~100%）
        crop_ratio: Tuple[float, float] = (0.6, 1.0),
        min_crop_sec: float = 0.8,      # 最短保留 0.8s（若原始更短则不裁剪）
        max_crop_sec: float = 6.0,      # 最长保留 6s（结合均值 5.66s）
        # 验证/测试：对超长样本做居中裁剪
        eval_max_sec: float = 6.0,
        train: bool = True,
        aug_prob: float = 0.5,
        dual_view: bool = True,  # ★ 同时产出整段和短切片
        short_min_sec: float = 0.4,
        short_max_sec: float = 1.2,
        noise_aug_prob: float = 0.3,  # ★ 见第3节“噪声强化”
        noise_snr_db: Tuple[float, float] = (10.0, 30.0),
        preemph: float = 0.95,
        highpass_prob: float = 0.3,
        highpass_hz: Tuple[int, int] = (800, 3000)
    ):
        self.records = []
        self.sr = sr
        self.crop_ratio = crop_ratio
        self.min_crop_samples = int(min_crop_sec * sr)
        self.max_crop_samples = int(max_crop_sec * sr)
        self.eval_max_samples = int(eval_max_sec * sr)
        self.train = train
        self.aug_prob = aug_prob
        self.dual_view = dual_view
        self.short_min_samples = int(short_min_sec * sr)
        self.short_max_samples = int(short_max_sec * sr)
        self.noise_aug_prob = noise_aug_prob
        self.noise_snr_db = noise_snr_db
        self.preemph = preemph
        self.highpass_prob = highpass_prob
        self.highpass_hz = highpass_hz
        # 将多份 csv 合并
        for csv_p, root in zip(csv_paths, roots):
            with open(csv_p, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    utt = row["utt"].strip()
                    rel = row["path"].strip()
                    label_str = row.get("label", "Bonafide").strip()
                    abspath = os.path.join(os.path.dirname(csv_p), rel)
                    # 如果 csv 在压缩包外单独放置，也支持 root+rel
                    if not os.path.exists(abspath):
                        abspath = os.path.join(root, rel)
                    if not os.path.exists(abspath):
                        # 再兜底找一层 part 目录
                        abspath = os.path.join(root, os.path.basename(os.path.dirname(csv_p)), rel)
                    if not os.path.exists(abspath):
                        # 允许文件缺失，跳过
                        continue
                    label_id = LABEL2ID[label_str] if label_str in LABEL2ID else None
                    self.records.append((utt, abspath, label_id))

        if len(self.records) == 0:
            raise RuntimeError("未找到任何音频记录，请检查 csv_paths 与 roots。")
    def _pre_emphasis(self, wav: np.ndarray, coef: float = 0.95) -> np.ndarray:
        if coef <= 0: return wav
        y = np.append(wav[0], wav[1:] - coef * wav[:-1])
        return np.clip(y.astype(np.float32), -1.0, 1.0)

    def _additive_noise(self, wav: np.ndarray, snr_db: float) -> np.ndarray:
        if snr_db <= 0: return wav
        sig_pwr = np.mean(wav**2) + 1e-12
        noise_pwr = sig_pwr / (10 ** (snr_db / 10.0))
        noise = np.random.randn(len(wav)).astype(np.float32) * np.sqrt(noise_pwr)
        y = wav + noise
        return np.clip(y, -1.0, 1.0)

    def _highpass(self, wav: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
        if not _HAS_TORCHAUDIO:
            # 简单一阶高通近似
            alpha = np.exp(-2*np.pi*cutoff/sr)
            y = np.zeros_like(wav, dtype=np.float32)
            for i in range(1, len(wav)):
                y[i] = wav[i] - wav[i-1] + alpha * y[i-1]
            return np.clip(y, -1.0, 1.0)
        else:
            x = torch.from_numpy(wav).float().unsqueeze(0)  # [1,T]
            y = torchaudio.functional.highpass_biquad(x, sample_rate=sr, cutoff_freq=cutoff)
            return y.squeeze(0).numpy().astype(np.float32)

    def _noise_augment(self, wav: np.ndarray) -> np.ndarray:
        if random.random() >= self.noise_aug_prob:
            return wav
        # 1) pre-emphasis
        y = self._pre_emphasis(wav, coef=self.preemph)
        # 2) random highpass
        if random.random() < self.highpass_prob:
            cut = np.random.uniform(self.highpass_hz[0], self.highpass_hz[1])
            y = self._highpass(y, sr=self.sr, cutoff=cut)
        # 3) additive noise with random SNR
        snr = np.random.uniform(self.noise_snr_db[0], self.noise_snr_db[1])
        y = self._additive_noise(y, snr_db=snr)
        return y

    def _short_crop(self, wav: np.ndarray) -> np.ndarray:
        L = len(wav)
        if L <= 0:
            return wav
        tgt = np.random.randint(self.short_min_samples, max(self.short_min_samples+1, min(self.short_max_samples, L)+1))
        if tgt >= L:
            return wav
        s = np.random.randint(0, L - tgt + 1)
        return wav[s: s + tgt]

    def __len__(self):
        return len(self.records)

    def _random_crop(self, wav: np.ndarray) -> np.ndarray:
        if self.max_samples is None or len(wav) <= self.max_samples:
            return wav
        start = np.random.randint(0, len(wav) - self.max_samples + 1)
        return wav[start: start + self.max_samples]
    def _ratio_crop(self, wav: np.ndarray) -> np.ndarray:
        """训练阶段：随机保留原片段的 r∈[r_min,r_max] 比例，再用秒级上下限夹住。"""
        L = len(wav)
        if L <= 0:
            return wav
        r_min, r_max = self.crop_ratio
        r = np.random.uniform(r_min, r_max)
        target = int(L * r)
        target = int(np.clip(target, self.min_crop_samples, self.max_crop_samples))
        if target >= L or target <= 0:
            return wav  # 太短或无需裁剪
        start = np.random.randint(0, L - target + 1)
        return wav[start: start + target]

    def _rand_gain(self, wav: np.ndarray, low=-6.0, high=6.0) -> np.ndarray:
        if random.random() < self.aug_prob:
            db = np.random.uniform(low, high)
            wav = wav * (10 ** (db / 20.0))
            wav = np.clip(wav, -1.0, 1.0)
        return wav

    def __getitem__(self, idx):
        utt, path, label_id = self.records[idx]
        wav_full = load_wav(path, target_sr=self.sr)

        if self.train:
            # 整段视角
            wav1 = self._ratio_crop(wav_full)
            wav1 = self._rand_gain(wav1)
            wav1 = self._noise_augment(wav1)  # 第3节新增的噪声强化
            # 短切片视角
            if self.dual_view:
                wav2 = self._short_crop(wav_full)
                wav2 = self._rand_gain(wav2)
                wav2 = self._noise_augment(wav2)
        else:
            wav1 = wav_full
            if len(wav1) > self.eval_max_samples:
                s = (len(wav1) - self.eval_max_samples) // 2
                wav1 = wav1[s: s + self.eval_max_samples]
            wav2 = None

        feat1 = torch.from_numpy(wav_to_logmel(wav1, sr=self.sr, n_mels=64)).float()
        item = {
            "utt": utt, "path": path,
            "feat": feat1,
            "label": torch.tensor(label_id if label_id is not None else -1, dtype=torch.long)
        }

        if self.train and self.dual_view:
            feat2 = torch.from_numpy(wav_to_logmel(wav2, sr=self.sr, n_mels=64)).float()
            item["feat_short"] = feat2

        return item

def collate_fn(batch):
    feats = [b["feat"] for b in batch]
    feat_pad, lengths = pad_stack(feats, pad_value=0.0)

    labels = torch.stack([b["label"] for b in batch], dim=0)
    utts = [b["utt"] for b in batch]
    paths = [b["path"] for b in batch]
    out = {"utt": utts, "path": paths, "feat": feat_pad, "lengths": lengths, "label": labels}

    if "feat_short" in batch[0]:
        feats_s = [b["feat_short"] for b in batch]
        feat_s_pad, len_s = pad_stack(feats_s, pad_value=0.0)
        out["feat_short"] = feat_s_pad
        out["lengths_short"] = len_s
    return out


################################################################################
# 模型：CNN + BiGRU + 多头查询注意力池化 + 帧级 Top-K MIL 约束
################################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, pool=(2,1)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool) if pool else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.pool(x)
        return x

class MultiHeadQueryAttention(nn.Module):
    """
    多头查询注意力池化：学习 H 个 query，每个 query 在时间维上做注意力权重，
    适合“只有中间一段异常”的情形。
    输入： X [B, T, D], mask [B, T] (True 有效)
    输出： pooled [B, H*D], attn [B, H, T]
    """
    def __init__(self, d_in: int, num_heads: int = 4):
        super().__init__()
        self.h = num_heads
        self.d = d_in
        self.Q = nn.Parameter(torch.randn(num_heads, d_in))       # [H, D]
        self.Wk = nn.Linear(d_in, d_in, bias=True)
        self.Wv = nn.Linear(d_in, d_in, bias=True)

    def forward(self, X, mask: Optional[torch.Tensor] = None):
        # X: [B,T,D]
        K = torch.tanh(self.Wk(X))    # [B,T,D]
        V = self.Wv(X)                # [B,T,D]
        # 计算每个头的注意力 a_h,t = <q_h, K_t>/sqrt(D)
        # einsum: [H,D] x [B,T,D] -> [B,H,T]
        logits = torch.einsum("hd,btd->bht", self.Q, K) / math.sqrt(self.d)
        if mask is not None:
            # mask: True 有效；把无效位置置为 -inf
            mask_ = (~mask).unsqueeze(1).expand_as(logits)  # [B,1,T] -> [B,H,T]
            logits = logits.masked_fill(mask_, float("-inf"))
        attn = torch.softmax(logits, dim=-1)  # [B,H,T]
        # 加权求和
        pooled = torch.einsum("bht,btd->bhd", attn, V).contiguous()  # [B,H,D]
        pooled = pooled.view(pooled.size(0), -1)  # [B, H*D]
        return pooled, attn

class AIGCDetector(nn.Module):
    def __init__(self, n_mels: int = 64, cnn_width: int = 64, cnn_depth: int = 3,
                 gru_hidden: int = 256, gru_layers: int = 2,
                 num_heads: int = 4, drop: float = 0.2):
        super().__init__()
        # 动态构建 CNN 块（频率维池化，时间步不降采样）
        blocks = []
        in_ch = 1
        ch = cnn_width
        for d in range(cnn_depth):
            out_ch = ch if d == 0 else min(ch * 2, 512)  # 逐步加宽，上限 512
            blocks.append(ConvBlock(in_ch, out_ch, k=3, p=1, pool=(2,1)))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)
        self.cnn_out_ch = in_ch
        self.dropout = nn.Dropout(drop)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # BiGRU 可配置层数/隐藏维
        self.gru = nn.GRU(
            input_size=self.cnn_out_ch,
            hidden_size=gru_hidden // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True
        )
        d_model = gru_hidden

        self.frame_head = nn.Linear(d_model, 1)
        self.attnpool = MultiHeadQueryAttention(d_in=d_model, num_heads=num_heads)
        self.clip_head = nn.Sequential(
            nn.Linear(d_model * num_heads, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(d_model, 1)
        )


    def forward(self, feat: torch.Tensor, lengths: torch.Tensor):
        """
        feat: [B, M, T] log-mel
        lengths: [B] in #frames
        返回：
          clip_logit: [B,1]
          frame_logit: [B,T]  (对齐原 T（近似），供 MIL/可解释）
          attn: [B,H,T_att]   (注意：T_att 可能与原 T 相等，因为 CNN 没降采样 T)
        """
        B, M, T = feat.shape
        x = feat.unsqueeze(1)  # [B,1,M,T]
        x = self.cnn(x)        # -> [B,C, M', T]  （时间步未降）
        x = self.freq_pool(x)  # -> [B,C,1,T]
        x = x.squeeze(2)       # -> [B,C,T]
        x = self.dropout(x)

        # 变换为 [B,T,C]
        x = x.transpose(1, 2).contiguous()  # [B,T,C]
        attn_mask = mask_by_length(x, lengths)  # [B,T]

        # BiGRU
        # 为了 pack，按长度降序
        lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
        x_sorted = x.index_select(0, idx_sort)
        packed = nn.utils.rnn.pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)
        out_packed, _ = self.gru(packed)
        out_seq, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # [B,T,C]（按最长对齐）
        # 恢复原顺序
        _, idx_unsort = torch.sort(idx_sort)
        out_seq = out_seq.index_select(0, idx_unsort)
        # pad 到原 T
        if out_seq.size(1) < T:
            pad = torch.zeros(B, T - out_seq.size(1), out_seq.size(2), device=out_seq.device)
            out_seq = torch.cat([out_seq, pad], dim=1)

        # 帧级 logit
        frame_logit = self.frame_head(out_seq).squeeze(-1)  # [B,T]

        # 注意力池化得到片段级表示
        pooled, attn = self.attnpool(out_seq, mask=attn_mask)    # [B,H*D], [B,H,T]
        clip_logit = self.clip_head(pooled)  # [B,1]
        return clip_logit, frame_logit, attn, attn_mask

################################################################################
# 损失函数：片段 BCE + Top-K MIL + 注意力熵约束
################################################################################

def attention_entropy_loss(attn: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    attn: [B,H,T]，mask: [B,T] True=有效
    计算每个头、每个样本的注意力分布熵，并做均值。鼓励更“尖”的注意力（低熵）。
    """
    B, H, T = attn.shape
    mask_h = mask.unsqueeze(1).expand(B, H, T)
    attn = attn + (~mask_h).float() * 0.0
    # 防止 log(0)
    attn_clamped = torch.clamp(attn, min=eps)
    ent = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)  # [B,H]
    # 对有效位置数量做归一化（近似）：log(#valid_T)
    valid_T = mask.sum(dim=-1, keepdim=True).clamp(min=1).float()  # [B,1]
    ent = ent / torch.log(valid_T + eps)  # 归一化到 [0,1]
    return ent.mean()

def topk_mil_loss(frame_logit: torch.Tensor,
                  lengths: torch.Tensor,
                  labels: torch.Tensor,
                  k_ratio: float = 0.1):
    """
    Mixed-precision safe Top-K MIL loss (use logits + BCEWithLogits)
    frame_logit: [B, T]  —— raw logits（未过 sigmoid）
    lengths:    [B]
    labels:     [B]  —— 0/1（int 或 float 都可）
    """
    B, T = frame_logit.shape
    losses = []
    for b in range(B):
        L = int(lengths[b].item())
        if L <= 0:
            continue
        k = max(1, int(math.ceil(L * k_ratio)))
        # 直接在 logits 上取 Top-K（与在概率上取 Top-K 等价，单调变换）
        topk_vals, _ = torch.topk(frame_logit[b, :L], k, largest=True, sorted=False)
        mean_logit = topk_vals.mean()                     # 标量 logits
        target = labels[b].float().view(1)                # [1]
        # 使用 logits 版本的 BCE，autocast 安全
        loss_b = F.binary_cross_entropy_with_logits(mean_logit.view(1), target)
        losses.append(loss_b)

    if len(losses) == 0:
        return torch.tensor(0.0, device=frame_logit.device)
    return torch.stack(losses).mean()

################################################################################
# 训练/验证/推理
################################################################################

def compute_metrics_from_scores(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """
    将分数阈值化为标签并统计指标。
    注意：任务说明将 Spoof 视为正类！
    """
    y_pred = (y_score >= thr).astype(np.int64)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    # 按题面给出的 F1 定义（F1 = Precision * 100%）
    f1_defined = precision * 100.0
    # 同时返回标准 F1 以供参考（不参与官方评分）
    f1_std = 2 * precision * recall / (precision + recall + 1e-12)
    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                Precision=precision, Recall=recall,
                F1_defined=f1_defined, F1_std=f1_std)

def build_dataloaders(
    train_csvs: List[str], train_roots: List[str],
    val_csvs: Optional[List[str]], val_roots: Optional[List[str]],
    batch_size: int = 16, num_workers: int = 4,
    # 裁剪参数
    crop_ratio: Tuple[float, float] = (0.6, 1.0),
    min_crop_sec: float = 0.8,
    max_crop_sec: float = 6.0,
    eval_max_sec: float = 6.0,
    # 如需回到“按类加权采样”，把它设为 True（默认 False，便于 epoch 级打乱且每样本仅出现一次）
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    ds_train = AIGCDataset(
        train_csvs, train_roots, train=True,
        crop_ratio=crop_ratio, min_crop_sec=min_crop_sec, max_crop_sec=max_crop_sec,
        eval_max_sec=eval_max_sec
    )

    if use_weighted_sampler:
        # 仅当你确实想用“按类加权采样”再开启；否则走 shuffle=True 更符合“打乱顺序”的需求
        labels = [r[2] for r in ds_train.records]
        class_sample_count = np.array([labels.count(0), labels.count(1)], dtype=np.float32) + 1e-6
        weights = [1.0 / class_sample_count[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler,
                              shuffle=False,  # sampler 与 shuffle 互斥
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    else:
        # 关键：shuffle=True -> 每个 epoch 都会随机打乱一次，且不重复抽样
        dl_train = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=True, persistent_workers=True, prefetch_factor=4
        )

    dl_val = None
    if val_csvs and val_roots:
        ds_val = AIGCDataset(
            val_csvs, val_roots, train=False,
            crop_ratio=crop_ratio, min_crop_sec=min_crop_sec, max_crop_sec=max_crop_sec,
            eval_max_sec=eval_max_sec
        )
        # 验证同样打乱（你要求验证集顺序也随机）
        if ds_val is not None:
            dl_val = DataLoader(
                ds_val, batch_size=batch_size, shuffle=True,  # 若想稳定评估可改 False
                num_workers=max(1, num_workers // 2), collate_fn=collate_fn,
                pin_memory=True, persistent_workers=True, prefetch_factor=2
            )

    return dl_train, dl_val

def train(
    train_csvs: List[str], train_roots: List[str],
    val_csvs: Optional[List[str]] = None, val_roots: Optional[List[str]] = None,
    save_path: str = "aigc_attn_model.pt",
    init_model_path: Optional[str] = None,
    epochs: int = 5, batch_size: int = 32, lr: float = 2e-4, weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    crop_ratio: Tuple[float, float] = (0.6, 1.0),
    min_crop_sec: float = 0.8,
    max_crop_sec: float = 6.0,
    eval_max_sec: float = 6.0,
    alpha_topk: float = 0.5, beta_ent: float = 0.05,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,   # 新增：需要时可开启
    lambda_short: float = 0.5
):
    set_seed(2025)
    dl_train, dl_val = build_dataloaders(
        train_csvs, train_roots, val_csvs, val_roots,
        batch_size=batch_size, num_workers=num_workers,
        crop_ratio=crop_ratio, min_crop_sec=min_crop_sec, max_crop_sec=max_crop_sec,
        eval_max_sec=eval_max_sec, use_weighted_sampler=use_weighted_sampler
    )

    model = AIGCDetector(n_mels=64, cnn_width=96,
                         cnn_depth=4, gru_hidden=384,
                         gru_layers=3, num_heads=6, drop=0.3).to(device)
    if init_model_path and os.path.exists(init_model_path):
        state = torch.load(init_model_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"[train] Loaded init weights from {init_model_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        running = []

        # —— 进度统计 —— #
        total_train = len(dl_train.dataset)
        seen = 0
        report_every = 1000   # 每处理 1000 条音频打印一次
        next_report = report_every

        for it, batch in enumerate(dl_train, 1):
            feat = batch["feat"].to(device)          # [B,M,T]
            lengths = batch["lengths"].to(device)     # [B]
            labels = batch["label"].to(device).float().unsqueeze(1)  # [B,1]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                clip_logit, frame_logit, attn, attn_mask = model(feat, lengths)
                loss_clip = bce(clip_logit, labels)
                loss_topk = topk_mil_loss(frame_logit, lengths, labels.squeeze(1).long(), k_ratio=0.1)
                loss_ent  = attention_entropy_loss(attn, attn_mask)
                loss = loss_clip + alpha_topk * loss_topk + beta_ent * loss_ent
                # ★ 双视角：对短切片也算一遍并加权
                if "feat_short" in batch:
                    feat_s = batch["feat_short"].to(device)
                    len_s  = batch["lengths_short"].to(device)
                    clip_logit_s, frame_logit_s, attn_s, attn_mask_s = model(feat_s, len_s)
                    loss_clip_s = bce(clip_logit_s, labels)  # 同一标签
                    loss_topk_s = topk_mil_loss(frame_logit_s, len_s, labels.squeeze(1).long(), k_ratio=0.2)
                    loss_ent_s  = attention_entropy_loss(attn_s, attn_mask_s)
                    loss = loss + lambda_short * (loss_clip_s + alpha_topk * loss_topk_s + beta_ent * loss_ent_s)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running.append(float(loss.detach().cpu().item()))

            # —— 进度打印：每 1000 条音频输出一次 —— #
            bs = feat.size(0)
            seen += bs
            if seen >= next_report or seen >= total_train:
                pct = 100.0 * seen / max(1, total_train)
                # 取当前 lr（Cosine 调度器）
                try:
                    cur_lr = scheduler.get_last_lr()[0]
                except Exception:
                    cur_lr = optimizer.param_groups[0]["lr"]
                print(f"[Epoch {ep}] progress {seen}/{total_train} ({pct:.1f}%) | last_loss={loss.item():.4f} | lr={cur_lr:.2e}")
                next_report += report_every

        scheduler.step()
        avg_train = float(np.mean(running)) if running else 0.0

        # —— 验证 —— #
        if dl_val is not None:
            model.eval()
            val_losses, all_labels, all_scores = [], [], []
            with torch.no_grad():
                for batch in dl_val:
                    feat = batch["feat"].to(device)
                    lengths = batch["lengths"].to(device)
                    labels = batch["label"].to(device).float().unsqueeze(1)

                    clip_logit, frame_logit, attn, attn_mask = model(feat, lengths)
                    loss_clip = bce(clip_logit, labels)
                    loss_topk = topk_mil_loss(frame_logit, lengths, labels.squeeze(1).long(), k_ratio=0.1)
                    loss_ent  = attention_entropy_loss(attn, attn_mask)
                    loss = loss_clip + alpha_topk * loss_topk + beta_ent * loss_ent

                    val_losses.append(float(loss.detach().cpu().item()))
                    score = torch.sigmoid(clip_logit).squeeze(1).cpu().numpy()
                    all_scores.append(score)
                    all_labels.append(labels.squeeze(1).cpu().numpy())

            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            y_true = np.concatenate(all_labels, axis=0).astype(np.int64)
            y_score = np.concatenate(all_scores, axis=0)
            metrics = compute_metrics_from_scores(y_true, y_score, thr=0.5)

            print(f"[Epoch {ep}] train_loss={avg_train:.4f} | val_loss={val_loss:.4f} "
                  f"| P={metrics['Precision']:.4f} R={metrics['Recall']:.4f} "
                  f"| F1(std)={metrics['F1_std']:.4f} | F1(def)={metrics['F1_defined']:.2f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save(best_state, save_path)
        else:
            # 没有验证集的兜底分支（你现在全局模式会有验证集，一般不会走到这里）
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, save_path)
            print(f"[Epoch {ep}] train_loss={avg_train:.4f} | saved to {save_path}")

    if best_state is not None:
        torch.save(best_state, save_path)
        print(f"训练完成，保存最优模型到：{save_path}")

    return save_path


@torch.no_grad()
def predict_scores_for_batch(model: nn.Module, feat: torch.Tensor, lengths: torch.Tensor, device: str):
    model.eval()
    feat = feat.to(device)
    lengths = lengths.to(device)
    clip_logit, frame_logit, attn, attn_mask = model(feat, lengths)
    score = torch.sigmoid(clip_logit).squeeze(1)  # [B]
    return score.cpu().numpy(), frame_logit.cpu().numpy(), attn.cpu().numpy()

@torch.no_grad()
def infer_to_csv(
    model_path: str,
    test_csvs: List[str],
    test_roots: List[str],
    out_csv: str = "result.csv",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_sec: float = 12.0,
    thr: float = 0.5
):
    ds = AIGCDataset(test_csvs, test_roots, train=False, eval_max_sec=max_sec)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model = AIGCDetector().to(device)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    for batch in dl:
        feat, lengths = batch["feat"], batch["lengths"]
        scores, _, _ = predict_scores_for_batch(model, feat, lengths, device)
        preds = (scores >= thr).astype(np.int64)
        for utt, path, sc, pd in zip(batch["utt"], batch["path"], scores, preds):
            rows.append({
                "utt": utt,
                "path": os.path.basename(path),
                "label": ID2LABEL[int(pd)],   # 输出标签
                "score": f"{sc:.6f}"          # 便于调试，可留可去（如果比赛只允许三列就别写）
            })

    # 写 result.csv（严格按题面只要三列的话，去掉 score 字段）
    fieldnames = ["utt", "path", "label"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"推理完成，结果保存：{out_csv}")
    return out_csv

def read_csv_labels(csv_paths: List[str], roots: List[str]) -> Dict[str, int]:
    """读取真值：返回 {utt: label_id}"""
    gt = {}
    for csv_p, _ in zip(csv_paths, roots):
        with open(csv_p, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", None)
                if label is None:
                    continue
                utt = row["utt"].strip()
                gt[utt] = LABEL2ID[label]
    return gt

def read_pred_labels(result_csv: str) -> Dict[str, int]:
    """读取预测：返回 {utt: label_id}"""
    pred = {}
    with open(result_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt = row["utt"].strip()
            label = row["label"].strip()
            pred[utt] = LABEL2ID[label]
    return pred

def evaluate_predictions(
    result_csv: str,
    gt_csvs: List[str],
    gt_roots: List[str],
    verbose: bool = True
) -> Dict[str, float]:
    """
    评分：统计 TP/TN/FP/FN 与 Precision/Recall/F1（题面定义）
    注意：以 Spoof 作为正类！
    """
    y_true_map = read_csv_labels(gt_csvs, gt_roots)
    y_pred_map = read_pred_labels(result_csv)

    common_utts = sorted(set(y_true_map.keys()) & set(y_pred_map.keys()))
    y_true = np.array([y_true_map[u] for u in common_utts], dtype=np.int64)
    y_pred = np.array([y_pred_map[u] for u in common_utts], dtype=np.int64)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1_defined = precision * 100.0  # 按题面定义
    f1_std = 2 * precision * recall / (precision + recall + 1e-12)

    if verbose:
        print(f"Total={len(common_utts)} | TP={TP} TN={TN} FP={FP} FN={FN}")
        print(f"Precision={precision:.6f} | Recall={recall:.6f} | F1(defined)={f1_defined:.2f} | F1(std)={f1_std:.6f}")

    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                Precision=precision, Recall=recall,
                F1_defined=f1_defined, F1_std=f1_std)

################################################################################
# 使用示例（请按你本地路径修改）
################################################################################

if __name__ == "__main__":
    """
    1) 训练（示例）：
    python aigc_detector.py train \
        --train_csvs ./DATA/aigc_speech_detection_tasks_part0/labels.csv \
        --train_roots ./DATA/aigc_speech_detection_tasks_part0 \
        --val_csvs   ./DATA/aigc_speech_detection_tasks_part1/labels.csv \
        --val_roots  ./DATA/aigc_speech_detection_tasks_part1 \
        --epochs 10 --batch_size 32 --save_path model.pt

    2) 推理生成 result.csv：
    python aigc_detector.py infer \
        --model_path model.pt \
        --test_csvs ./DATA/aigc_speech_detection_tasks_part2/labels.csv \
        --test_roots ./DATA/aigc_speech_detection_tasks_part2 \
        --out_csv result.csv

    3) 评估（使用带真值的 csv 对 result.csv 打分）：
    python aigc_detector.py eval \
        --result_csv result.csv \
        --gt_csvs ./DATA/aigc_speech_detection_tasks_part2/labels.csv \
        --gt_roots ./DATA/aigc_speech_detection_tasks_part2
    """
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train")
    p_train.add_argument("--train_csvs", nargs="+", required=True)
    p_train.add_argument("--train_roots", nargs="+", required=True)
    p_train.add_argument("--val_csvs", nargs="+", default=None)
    p_train.add_argument("--val_roots", nargs="+", default=None)
    p_train.add_argument("--save_path", type=str, default="aigc_attn_model.pt")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.add_argument("--max_sec", type=float, default=12.0)
    p_train.add_argument("--alpha_topk", type=float, default=0.5)
    p_train.add_argument("--beta_ent", type=float, default=0.05)
    p_train.add_argument("--num_workers", type=int, default=4)

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--model_path", type=str, required=True)
    p_infer.add_argument("--test_csvs", nargs="+", required=True)
    p_infer.add_argument("--test_roots", nargs="+", required=True)
    p_infer.add_argument("--out_csv", type=str, default="result.csv")
    p_infer.add_argument("--batch_size", type=int, default=32)
    p_infer.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p_infer.add_argument("--max_sec", type=float, default=12.0)
    p_infer.add_argument("--thr", type=float, default=0.5)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--result_csv", type=str, required=True)
    p_eval.add_argument("--gt_csvs", nargs="+", required=True)
    p_eval.add_argument("--gt_roots", nargs="+", required=True)
    p_train.add_argument("--init_model_path", type=str, default=None)
    p_train.add_argument("--crop_ratio", nargs=2, type=float, default=[0.6, 1.0])
    p_train.add_argument("--min_crop_sec", type=float, default=0.8)
    p_train.add_argument("--max_crop_sec", type=float, default=6.0)
    p_train.add_argument("--eval_max_sec", type=float, default=6.0)
    args = parser.parse_args()

    if args.cmd == "train":
        train(
            train_csvs=args.train_csvs, train_roots=args.train_roots,
            val_csvs=args.val_csvs, val_roots=args.val_roots,
            save_path=args.save_path, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay, device=args.device,
            max_sec=args.max_sec, alpha_topk=args.alpha_topk, beta_ent=args.beta_ent,
            num_workers=args.num_workers,
            init_model_path=args.init_model_path,
            crop_ratio=tuple(args.crop_ratio),
            min_crop_sec=args.min_crop_sec,
            max_crop_sec=args.max_crop_sec,
            eval_max_sec=args.eval_max_sec
        )
    elif args.cmd == "infer":
        infer_to_csv(
            model_path=args.model_path,
            test_csvs=args.test_csvs, test_roots=args.test_roots,
            out_csv=args.out_csv, batch_size=args.batch_size,
            device=args.device, max_sec=args.max_sec, thr=args.thr
        )
    elif args.cmd == "eval":
        evaluate_predictions(
            result_csv=args.result_csv,
            gt_csvs=args.gt_csvs, gt_roots=args.gt_roots,
            verbose=True
        )
    else:
        parser.print_help()
