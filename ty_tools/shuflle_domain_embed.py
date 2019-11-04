import os
import argparse
import random

def shuffle_list(source_dir, target_dir):
    if not os.path.exists(source_dir):
        raise AssertionError

    if not os.path.exists(target_dir):
        raise AssertionError

    all_source_files = [a for a in os.listdir(target_dir) if a.endswith('.txt')]

    for f in all_source_files:
        os.remove(os.path.join(target_dir, f))

    all_list = [a for a in os.listdir(source_dir) if a.endswith('.txt')]

    num_of_file = len(all_list)

    t_list = list(range(num_of_file))

    random.shuffle(t_list)

    for i, s in enumerate(all_list):
        t = t_list[i]
        s_f = os.path.join(source_dir, s)
        t_f = os.path.join(target_dir, all_list[t])
        print('Linking source:{} target:{}'.format(s_f, t_f))
        os.link(s_f, t_f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--p', type=str, default=None)
    parser.add_argument('--t', type=str, default=None)

    args = parser.parse_args()

    shuffle_list(args.p, args.t)


