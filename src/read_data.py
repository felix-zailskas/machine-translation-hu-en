import re
import csv
import pandas as pd

sample_proportion = 0.1
offset = 1500
english_path = "data/hu-en/europarl-v7.hu-en.en"
hungarian_path = "data/hu-en/europarl-v7.hu-en.hu"


def raw_to_csv(path1: str, path2: str, sample_proportion: float, offset: int):
    lines_f1, lines_f2 = [], []
    with open(path1, "r") as f1, open(path2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        assert len(lines1) == len(lines2)
        needed_lines = int(len(lines1) * sample_proportion)
        read_lines = 0
        for l1, l2 in zip(lines1[offset:], lines2[offset:]):
            if read_lines == needed_lines:
                break
            if not (l1[0].isalpha() and l2[0].isalpha()):
                continue
            lines_f1.append(l1.strip())
            lines_f2.append(l2.strip())
            read_lines += 1
    rows = zip(lines_f1, lines_f2)
    with open("data/sampled_data.csv", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


raw_to_csv(english_path, hungarian_path, sample_proportion, offset)

# df = pd.read_csv("data/sampled_data.csv")
# print(df.head())
