import argparse

import os

def parse_file(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()
        performance_str = lines[-2]

    return performance_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default=None)

    args = parser.parse_args()

    file_path = os.path.join(args.p, 'log.txt')
    if not os.path.exists(file_path):
        raise AssertionError

    peformance_str = parse_file(file_path)
    print(peformance_str)

    with open('shuflle_result.txt', 'a') as f:
        f.write(peformance_str + '\n')
