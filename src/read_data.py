import re
import csv
import pandas as pd

sample_proportion = 0.1
offset = 1500
english_path = "data/hu-en/europarl-v7.hu-en.en"
hungarian_path = "data/hu-en/europarl-v7.hu-en.hu"

def read_lines(path : str) -> list:
    with open(path, "r") as f:
        data = [line.strip() for line in f if line[0].isalpha()]
    return data

def sample_data(data : list, sample_proprtion : float, offset : int, ) -> list: 
    return data[offset:(offset + int(sample_proportion*len(data)))]

def raw_to_csv(path1 : str, path2 : str, sample_proportion : float, offset : int): 
    lines1 = read_lines(path1)
    lines2 = read_lines(path2)
    data1 = sample_data(lines1, sample_proportion, offset)
    data2 = sample_data(lines2, sample_proportion, offset)
    rows = zip(data1, data2)
    with open("sampled_data", "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

raw_to_csv(english_path, hungarian_path, sample_proportion, offset)

df = pd.read_csv("sampled_data")
print(df.head())