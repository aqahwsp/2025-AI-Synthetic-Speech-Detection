# -*- coding: utf-8 -*-
"""
check_csv_formats.py
遍历 ./DATA/aigc_speech_detection_tasks_part*/ 下的 CSV，
检测列名是否规范（utt/path/label），并报告别名、缺失、额外列等。
可选 --write-normalized 生成规范化 CSV（仅含 utt,path,label，自动从别名映射）。
"""
import os
import re
import csv
import argparse
from collections import Counter, defaultdict

def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

# 默认别名映射（规范列 -> 可能出现的别名列表，大小写不敏感）
DEFAULT_ALIASES = {
    "utt":   ["utt", "utt_id", "uttid", "id", "uid", "sample_id"],
    "path":  ["path", "wav_path", "wav", "file", "file_path", "filepath", "relpath", "audio", "audio_path"],
    "label": ["label", "y", "class", "target"],
}

EXPECTED = ["utt", "path", "label"]

def list_part_dirs(root: str):
    pat = re.compile(r"^aigc_speech_detection_tasks_part\d+$")
    dirs = [d for d in os.listdir(root) if pat.match(d) and os.path.isdir(os.path.join(root, d))]
    dirs.sort(key=natural_key)
    return [os.path.join(root, d) for d in dirs]

def find_csvs_in_dir(d: str):
    return sorted([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".csv")], key=natural_key)

def normalize_header(h):
    # 去掉 BOM/空白，统一小写
    return h.replace("\ufeff", "").strip().lower()

def build_reverse_alias_map(aliases):
    rev = {}
    for canonical, alist in aliases.items():
        for a in alist:
            rev[normalize_header(a)] = canonical
    return rev

def analyze_csv(csv_path: str, aliases=DEFAULT_ALIASES, sample_check: int = 100):
    rev_alias = build_reverse_alias_map(aliases)
    folder = os.path.dirname(csv_path)

    # 读取
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:  # utf-8-sig 处理 BOM
        reader = csv.DictReader(f)
        raw_headers = reader.fieldnames or []
        norm_headers = [normalize_header(h) for h in raw_headers]

        # 规范列匹配（直接或通过别名）
        col_map = {}   # 规范列 -> 实际列名（原始）
        extras = []    # 既不是期望列也不是其别名的列（保留原名）
        seen_norm = set()
        for raw, norm in zip(raw_headers, norm_headers):
            seen_norm.add(norm)
            if norm in rev_alias:
                canon = rev_alias[norm]  # 映射到规范名
                # 若已匹配到一个同名，保留第一个
                if canon not in col_map:
                    col_map[canon] = raw
            else:
                extras.append(raw)

        missing = [c for c in EXPECTED if c not in col_map]

        # 统计标签与抽样检查路径存在性
        label_counter = Counter()
        n_rows = 0
        n_missing_vals = defaultdict(int)
        path_missing_fs = 0
        sample_checked = 0

        # 为了避免 5w 行都做磁盘访问，只抽样前 sample_check 条做文件存在性检查
        for row in reader:
            n_rows += 1
            # 拿到规范后的值
            utt = row.get(col_map.get("utt", ""), "").strip()
            path_val = row.get(col_map.get("path", ""), "").strip()
            label_val = row.get(col_map.get("label", ""), "").strip()

            if utt == "":   n_missing_vals["utt"] += 1
            if path_val == "": n_missing_vals["path"] += 1
            if label_val == "": n_missing_vals["label"] += 1

            if label_val != "":
                label_counter[label_val] += 1

            # 抽样检查路径存在性
            if sample_checked < sample_check and path_val:
                # CSV 一般是相对路径：与 csv 同目录拼接
                abs_path = os.path.join(folder, path_val)
                if not os.path.exists(abs_path):
                    path_missing_fs += 1
                sample_checked += 1

        report = {
            "csv": csv_path,
            "rows": n_rows,
            "raw_headers": raw_headers,
            "normalized_headers": norm_headers,
            "resolved_columns": col_map,       # 规范列 -> 实际列
            "missing_expected_columns": missing,
            "extra_columns": [e for e in extras if normalize_header(e) not in rev_alias],
            "label_distribution": dict(label_counter),
            "empty_values": dict(n_missing_vals),
            "path_exist_check": {
                "sample_size": sample_checked,
                "missing_count": path_missing_fs,
            },
            "suggestion": {
                "rename_map": {col_map.get(k, f"<MISSING:{k}>"): k for k in EXPECTED},
                "can_be_normalized": (set(missing) == set()),  # 没缺必须列就能规范化
            },
        }
        return report

def print_report(rep):
    print("="*80)
    print(f"[CSV] {rep['csv']}")
    print(f"  Rows: {rep['rows']}")
    print(f"  Headers(raw): {rep['raw_headers']}")
    print(f"  Headers(norm): {rep['normalized_headers']}")
    print(f"  Resolved (canonical -> actual): {rep['resolved_columns']}")
    print(f"  Missing expected: {rep['missing_expected_columns']}")
    print(f"  Extra columns: {rep['extra_columns']}")
    print(f"  Label distribution: {rep['label_distribution']}")
    print(f"  Empty values: {rep['empty_values']}")
    pe = rep['path_exist_check']
    print(f"  Path exists (sample {pe['sample_size']}): missing {pe['missing_count']}")
    sug = rep['suggestion']
    print(f"  Suggest rename_map (actual->canonical): {sug['rename_map']}")
    print(f"  Can normalize: {sug['can_be_normalized']}")

def write_normalized_csv(rep, out_path: str):
    """
    按报告把 CSV 规范化为仅含三列：utt,path,label。
    当存在别名（例如 wav_path）时会自动重命名。
    """
    src = rep["csv"]
    col_map = rep["resolved_columns"]
    missing = rep["missing_expected_columns"]
    if missing:
        raise RuntimeError(f"{src} 缺少必要列，无法规范化：{missing}")

    with open(src, "r", newline="", encoding="utf-8-sig") as f_in, \
         open(out_path, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["utt", "path", "label"])
        writer.writeheader()
        for row in reader:
            out_row = {
                "utt":   row.get(col_map["utt"], "").strip(),
                "path":  row.get(col_map["path"], "").strip(),
                "label": row.get(col_map["label"], "").strip(),
            }
            writer.writerow(out_row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./DATA", help="根目录，下面是各个 part* 目录")
    ap.add_argument("--sample_check", type=int, default=100, help="每个 CSV 抽样检查路径存在性的条数")
    ap.add_argument("--write-normalized", action="store_true", help="为每个 CSV 写出 *_normalized.csv（仅含 utt,path,label）")
    args = ap.parse_args()

    parts = list_part_dirs(args.data_root)
    if not parts:
        print(f"[WARN] No part dirs found under {args.data_root}")
        return

    print(f"[INFO] Found {len(parts)} part dirs")
    for d in parts:
        csvs = find_csvs_in_dir(d)
        if not csvs:
            print(f"[WARN] No CSV in {d}")
            continue
        for cp in csvs:
            rep = analyze_csv(cp, aliases=DEFAULT_ALIASES, sample_check=args.sample_check)
            print_report(rep)
            if args.write_normalized and rep["suggestion"]["can_be_normalized"]:
                out = os.path.splitext(cp)[0] + "_normalized.csv"
                write_normalized_csv(rep, out)
                print(f"  -> Wrote normalized CSV: {out}")

if __name__ == "__main__":
    main()
