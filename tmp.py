import csv
from collections import Counter

def count_third_column_values(file_path):
    counts = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头，如果没有表头可以去掉这一行
        for row in reader:
            if len(row) >= 3:  # 确保至少有三列
                value = row[2].strip()  # 取第三列并去掉空格
                if value:  # 跳过空值
                    counts[value] += 1
    return counts

if __name__ == "__main__":
    file_path = "/202321633095/WSP/PatchBackdoor-main/Audiocheck/DATA/aigc_speech_detection_tasks_part11/aigc_speech_detection_tasks_part11.csv"  # 替换为你的CSV文件路径
    result = count_third_column_values(file_path)
    for value, count in result.items():
        print(f"{value}: {count}")
