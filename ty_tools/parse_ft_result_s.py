import os
import json


train_img_dir = '../datasets/bdd100k/data_part/time-weather-scene-part/images/train'
dirs = [
        "../results/ft_base/time-weather-scene-part/bn_post",
        "../results/ft_base/time-weather-scene-part/base_test",
        "../results/ft_base/time-weather-scene-part/simple",
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

    img_dir = os.path.join(train_img_dir, name)
    num_img = len(os.listdir(img_dir))
    log_strs.append("{}: Num of image {} \n".format(name, num_img))

    for dir_path, d_n in zip(dirs, dir_name):
        log_path = os.path.join(dir_path, name, 'log.txt')
        if not os.path.exists(log_path):
            continue

        p_str = parse_one_file(log_path)

        log_strs.append("{}: {}".format(d_n, p_str))

#print(log_strs)

with open('./ft_result.txt', 'w') as f:
    f.writelines(log_strs)
