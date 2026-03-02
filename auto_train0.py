# -*- coding: utf-8 -*-
"""
auto_train.py  (GLOBAL MIX FIX)
- 发现 ./DATA/aigc_speech_detection_tasks_part*/ 目录
- 读取目录内 CSV（列名可能为 wav_path/path），统一成 utt,path,label，并将 path 归一为 basename
- 可选：仅保留磁盘存在的 wav
- 对每个目录 7:3 切分 -> train_split.csv / test_split.csv
- 两种训练模式：
  1) global（默认）：将所有 part 的 train_split/test_split 聚合为一个“大训练集/大验证集”，一次性训练 -> 解决“目录级单类”问题
  2) sequential：旧逻辑，逐 part 持续训练（不推荐在你当前数据分布下使用）
- 支持 --exclude 屏蔽目录
"""
import os, re, csv, random, argparse, shutil
from typing import List, Dict, Tuple
from aigc_detector0 import train

# ============== 工具 ==============
def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def _normalize_names(names: List[str]) -> List[str]:
    norm, seen = [], set()
    for n in names or []:
        n = os.path.basename(n.strip().rstrip("/\\"))
        if n and n not in seen:
            seen.add(n); norm.append(n)
    return norm

def find_part_dirs(data_root: str, exclude_names: List[str]) -> Tuple[List[str], List[str]]:
    pat = re.compile(r"^aigc_speech_detection_tasks_part\d+$")
    all_dirs = [d for d in os.listdir(data_root) if pat.match(d) and os.path.isdir(os.path.join(data_root, d))]
    all_dirs.sort(key=natural_key)
    ex = set(_normalize_names(exclude_names))
    included, excluded = [], []
    for d in all_dirs:
        if d in ex:
            excluded.append(os.path.join(data_root, d))
        else:
            included.append(os.path.join(data_root, d))
    return included, excluded

def find_csv_in_dir(part_dir: str) -> str:
    csvs = [os.path.join(part_dir, f) for f in os.listdir(part_dir) if f.lower().endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No csv found in {part_dir}")
    csvs.sort(key=natural_key)
    return csvs[0]

# 列别名（小写）
ALIASES = {
    "utt":   ["utt", "utt_id", "uttid", "id", "uid", "sample_id"],
    "path":  ["path", "wav_path", "wav", "file", "file_path", "filepath", "relpath", "audio", "audio_path"],
    "label": ["label", "y", "class", "target"],
}
EXPECTED = ["utt", "path", "label"]

def _norm_header(h: str) -> str:
    return h.replace("\ufeff", "").strip().lower()

def resolve_columns(headers: List[str]) -> Dict[str, str]:
    """返回 规范名->实际列名 的映射，如 {'path': 'wav_path'}"""
    h_norm = { _norm_header(h): h for h in headers }
    col_map: Dict[str, str] = {}
    for canon, alias_list in ALIASES.items():
        for a in alias_list:
            if a in h_norm:
                col_map[canon] = h_norm[a]
                break
    missing = [k for k in EXPECTED if k not in col_map]
    if missing:
        raise RuntimeError(f"CSV 缺少必要列: {missing} | headers={headers}")
    return col_map

# ============== 读取 & 规范化 ==============
def read_and_normalize_rows(part_dir: str, csv_path: str, keep_only_existing: bool = True) -> List[Dict[str, str]]:
    """
    - 将列映射为 utt/path/label
    - path 归一为 basename（与 part_dir 直接拼）
    - 可选：仅保留磁盘存在的 wav
    """
    rows_norm: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"Empty header: {csv_path}")
        col_map = resolve_columns(reader.fieldnames)

        dropped, total = 0, 0
        for r in reader:
            total += 1
            utt   = (r.get(col_map["utt"],   "") or "").strip()
            p_raw = (r.get(col_map["path"],  "") or "").strip().replace("\\", "/")
            label = (r.get(col_map["label"], "") or "").strip()
            p_base = os.path.basename(p_raw)                         # 关键：只保留文件名
            abs_candidate = os.path.join(part_dir, p_base)
            if keep_only_existing and not os.path.exists(abs_candidate):
                dropped += 1
                continue
            rows_norm.append({"utt": utt, "path": p_base, "label": label})

    kept = len(rows_norm)
    print(f"[normalize] {os.path.basename(part_dir)}: kept {kept}/{total} rows "
          f"({'{:.2f}'.format(100.0*kept/max(1,total))}%), dropped {dropped}")
    if kept == 0:
        raise RuntimeError(f"{part_dir}: 所有样本均未在磁盘找到，请检查 wav 是否解压到该目录。")
    return rows_norm

# ============== 切分 & 写出 ==============
def stratified_split_norm(rows_norm: List[Dict[str,str]], ratio: float = 0.7, seed: int = 2025):
    random.seed(seed)
    pos = [r for r in rows_norm if r["label"] == "Spoof"]
    neg = [r for r in rows_norm if r["label"] == "Bonafide"]
    if len(pos) == 0 or len(neg) == 0:
        # 单类目录 -> 随机切分
        all_rows = rows_norm[:]
        random.shuffle(all_rows)
        k = int(len(all_rows) * ratio)
        return all_rows[:k], all_rows[k:]

    def split(arr):
        random.shuffle(arr)
        k = int(len(arr) * ratio)
        return arr[:k], arr[k:]
    tr_p, te_p = split(pos); tr_n, te_n = split(neg)
    train_rows = tr_p + tr_n; test_rows = te_p + te_n
    random.shuffle(train_rows); random.shuffle(test_rows)
    return train_rows, test_rows

def write_csv(path: str, rows: List[Dict[str,str]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPECTED)
        w.writeheader()
        w.writerows(rows)

# ============== 主流程 ==============
def auto_train(
    data_root: str = "./DATA",
    save_dir: str = "./checkpoints0",
    epochs: int = 4,
    batch_size: int = 32,
    device: str = "cuda",
    crop_ratio=(0.6, 1.0),
    min_crop_sec=0.8,
    max_crop_sec=6.0,
    eval_max_sec=6.0,
    exclude: List[str] = None,
    mode: str = "global",          # "global"（推荐）或 "sequential"
):
    os.makedirs(save_dir, exist_ok=True)
    parts, excluded = find_part_dirs(data_root, exclude or [])

    print(f"[exclude] Requested: {exclude or []}")
    if excluded:
        print("[exclude] Found & skipped:")
        for p in excluded: print("  -", os.path.basename(p))
    else:
        print("[exclude] No existing dirs matched to skip.")

    if not parts:
        raise RuntimeError(f"No part directories to train under {data_root}")

    print(f"[discover] {len(parts)} part dirs to train:")
    for p in parts: print("  -", os.path.basename(p))

    # 逐目录规范化 + 7:3 切分，同时累计“全局训练/验证列表”
    train_csvs_all, train_roots_all = [], []
    val_csvs_all,   val_roots_all   = [], []
    global_counts = {"train": {"Spoof":0, "Bonafide":0}, "val": {"Spoof":0, "Bonafide":0}}

    for part_dir in parts:
        csv_path = find_csv_in_dir(part_dir)
        print(f"[csv] using {os.path.basename(csv_path)}")
        rows_norm = read_and_normalize_rows(part_dir, csv_path, keep_only_existing=True)
        train_rows, val_rows = stratified_split_norm(rows_norm, ratio=0.7, seed=2025)

        # 写回每个目录内的切分文件（依旧保留）
        train_csv = os.path.join(part_dir, "train_split.csv")
        val_csv   = os.path.join(part_dir, "test_split.csv")
        write_csv(train_csv, train_rows)
        write_csv(val_csv, val_rows)
        print(f"[split] train:{len(train_rows)}  val:{len(val_rows)}  -> {os.path.basename(part_dir)}")

        # 统计类别数（用于打印汇总）
        for r in train_rows: global_counts["train"][r["label"]] += 1
        for r in val_rows:   global_counts["val"][r["label"]]   += 1

        # 聚合：把所有 part 的 train/val 都加入“全局列表”
        train_csvs_all.append(train_csv); train_roots_all.append(part_dir)
        val_csvs_all.append(val_csv);     val_roots_all.append(part_dir)

    print("[global] aggregated counts:")
    print(f"  train -> Bonafide={global_counts['train']['Bonafide']}  Spoof={global_counts['train']['Spoof']}")
    print(f"  val   -> Bonafide={global_counts['val']['Bonafide']}    Spoof={global_counts['val']['Spoof']}")

    if mode.lower() == "global":
        # —— 推荐：一次全量训练，训练/验证中都有两类样本 ——
        out_ckpt = os.path.join(save_dir, "model_global.pt")
        print(f"[train:global] epochs={epochs}  out={out_ckpt}")
        train(
            train_csvs=train_csvs_all, train_roots=train_roots_all,
            val_csvs=val_csvs_all,     val_roots=val_roots_all,
            save_path=out_ckpt,
            init_model_path=None,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            crop_ratio=crop_ratio,
            min_crop_sec=min_crop_sec,
            max_crop_sec=max_crop_sec,
            eval_max_sec=eval_max_sec,
            alpha_topk=0.5,
            beta_ent=0.05,
            num_workers=12,
        )
        final_ckpt = os.path.join(save_dir, "model_final.pt")
        if out_ckpt != final_ckpt:
            shutil.copyfile(out_ckpt, final_ckpt)
        print(f"[done] Final model saved at: {final_ckpt}")

    else:
        # —— 兼容旧逻辑：逐 part 持续训练（不推荐在当前数据分布下） ——
        init_ckpt = None
        for idx, part_dir in enumerate(parts):
            train_csv = os.path.join(part_dir, "train_split.csv")
            val_csv   = os.path.join(part_dir, "test_split.csv")
            out_ckpt = os.path.join(save_dir, f"model_part{idx}.pt")
            print(f"[train:sequential] part={os.path.basename(part_dir)} epochs={epochs} init={init_ckpt}")
            train(
                train_csvs=[train_csv], train_roots=[part_dir],
                val_csvs=[val_csv],     val_roots=[part_dir],
                save_path=out_ckpt,
                init_model_path=init_ckpt,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                crop_ratio=crop_ratio,
                min_crop_sec=min_crop_sec,
                max_crop_sec=max_crop_sec,
                eval_max_sec=eval_max_sec,
                alpha_topk=0.5,
                beta_ent=0.05,
                num_workers=12,
            )
            init_ckpt = out_ckpt
        final_ckpt = os.path.join(save_dir, "model_final.pt")
        if init_ckpt and init_ckpt != final_ckpt:
            shutil.copyfile(init_ckpt, final_ckpt)
        print(f"[done] Final model saved at: {final_ckpt}")

# ============== CLI ==============
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./DATA")
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--crop_ratio", nargs=2, type=float, default=[0.6, 1.0])
    ap.add_argument("--min_crop_sec", type=float, default=0.8)
    ap.add_argument("--max_crop_sec", type=float, default=6.0)
    ap.add_argument("--eval_max_sec", type=float, default=6.0)
    ap.add_argument("--exclude", nargs="*", default=[], help="屏蔽目录基名列表")
    ap.add_argument("--mode", type=str, default="global", choices=["global", "sequential"],
                    help="global: 汇总所有 part 一次训练（推荐）；sequential: 逐 part 持续训练（你的数据分布下不推荐）")
    args = ap.parse_args()

    auto_train(
        data_root=args.data_root,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        crop_ratio=tuple(args.crop_ratio),
        min_crop_sec=args.min_crop_sec,
        max_crop_sec=args.max_crop_sec,
        eval_max_sec=args.eval_max_sec,
        exclude=args.exclude,
        mode=args.mode,
    )
