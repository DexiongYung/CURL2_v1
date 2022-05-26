from ast import arg
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        default="./logs/cartpole_swingup/center_crop/seed_23/train_steps_100000/05-25/train.csv",
    )
    args = parser.parse_args()
    return args


def columns_correlation_coefficient(args):
    dir_path = os.path.dirname(os.path.realpath(args.file_path))
    file_name = os.path.basename(args.file_path)
    file_name_no_ext = file_name.split(".")[0]

    df = pd.read_csv(args.file_path)

    with open(os.path.join(dir_path, file_name_no_ext + "_correlation.txt"), "w") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(df.corr(method="pearson"))


if __name__ == "__main__":
    args = parse_args()
    columns_correlation_coefficient(args=args)
