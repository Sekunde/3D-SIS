import os
import argparse

mapping = {
        0:0,
        1:3,
        2:4,
        3:5,
        4:6,
        5:7,
        6:8,
        7:9,
        8:10,
        9:11,
        10:12,
        11:14,
        12:16,
        13:24,
        14:28,
        15:33,
        16:34,
        17:36,
        18:39
        }


parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', default="./ScanNet_Benchmark_Result")
opt = parser.parse_args()

def main():
    benchmark_path = opt.pred_path

    for txt_file in os.listdir(benchmark_path):
        if os.path.isdir(os.path.join(benchmark_path, txt_file)):
            continue
        print(txt_file)
        writelines = []

        lines = open(os.path.join(benchmark_path, txt_file)).readlines()
        for line in lines:
            line = line.split()
            line[1] = str(mapping[int(line[1])])
            writelines.append(' '.join(line) + '\n')
        f = open(os.path.join(benchmark_path, txt_file), 'w')
        f.writelines(writelines)
        f.close()

if __name__ == "__main__":
    main()


