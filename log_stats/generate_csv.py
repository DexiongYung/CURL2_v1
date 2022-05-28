from ast import arg
import os
import json
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        default="./logs/cartpole_swingup/center_crop/seed_23/train_steps_100000/05-25/train.log",
    )
    args = parser.parse_args()
    return args


def generate_csv_from_logs(args):
    data_dict = None
    dir_path = os.path.dirname(os.path.realpath(args.file_path))
    file_name = os.path.basename(args.file_path)
    file_name_no_ext = file_name.split(".")[0]

    with open(args.file_path) as f:
        f = f.readlines()

    for line in f:
        line_dict = json.loads(line)
        if "episode" in line_dict and line_dict["step"] > 1000:
            if data_dict is None:
                data_dict = dict()
                for key, val in line_dict.items():
                    data_dict[key] = [val]
            else:
                for key, val in data_dict.items():
                    data_dict[key].append(line_dict.get(key, None))
        else:
            continue

    pd.DataFrame.from_dict(data_dict).to_csv(
        os.path.join(dir_path, file_name_no_ext + ".csv")
    )


if __name__ == "__main__":
    args = parse_args()
    generate_csv_from_logs(args=args)
