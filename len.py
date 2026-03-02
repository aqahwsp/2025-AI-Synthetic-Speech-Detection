import os
import wave

# 你的音频文件目录
folder = "/202321633095/WSP/PatchBackdoor-main/Audiocheck/DATA/"

lengths = []

for root, _, files in os.walk(folder):
    for file in files:
        if file.lower().endswith(".wav"):
            filepath = os.path.join(root, file)
            try:
                with wave.open(filepath, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
                    lengths.append(duration)
            except wave.Error:
                print(f"无法读取 {filepath}")

if lengths:
    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    print(f"总文件数: {len(lengths)}")
    print(f"平均长度: {avg_length:.2f} 秒")
    print(f"最短长度: {min_length:.2f} 秒")
else:
    print("未找到有效的 WAV 文件。")
