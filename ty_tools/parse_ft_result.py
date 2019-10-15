import os
import json

dirs = ["/home/tyzhang/projects/maskrcnn-benchmark/results/ft_base/time-weather-part/baseline",
        "/home/tyzhang/projects/maskrcnn-benchmark/results/ft_base/time-weather-part/base_test",
        "/home/tyzhang/projects/maskrcnn-benchmark/results/ft_base/time-weather-part/9ep_2e-3lr",
        #"/home/tyzhang/projects/maskrcnn-benchmark/results/ft_base/time-weather-part/2e-3lr",
        ]

dir_name = [s.split('/')[-1] for s in dirs]


def parse_one_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        performance_str = lines[-2]

        print(performance_str)

    return performance_str

all_names = os.listdir(dirs[0])

log_strs = []

for name in all_names:
    log_strs.append("{}:\n".format(name))
    for dir_path, d_n in zip(dirs, dir_name):
        log_path = os.path.join(dir_path, name, 'log.txt')

        p_str = parse_one_file(log_path)

        log_strs.append("{}: {}".format(d_n, p_str))

#print(log_strs)

with open('./ft_result.txt', 'w') as f:
    f.writelines(log_strs)
