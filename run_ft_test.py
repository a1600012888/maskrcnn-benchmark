import os

part_name = "time-weather-scene-part"
NGPUS = 4
batch_size = 16 * NGPUS
num_epoch = 18
stpes = [12, 16]

base_cmd = "python3 -m torch.distributed.launch --nproc_per_node={} tools/train_net.py".format(NGPUS) +\
           " --config-file './configs/template/ft_test_self_base.yaml' "

img_dir = os.path.join("./datasets/bdd100k/data_part",
                       part_name, "images/train")

all_names = os.listdir(img_dir)
img_numbers = []
for name in all_names:
    target_dir = os.path.join(img_dir, name)
    num = len(os.listdir(target_dir))
    img_numbers.append(num)


for name, num in zip(all_names, img_numbers):
    if num < batch_size:
        continue
    self_str = "self_" + part_name + "_" + name

    iters_per_epoch = num // batch_size + 1

    solver_steps = [ep * iters_per_epoch for ep in stpes]
    solver_max_iter = num_epoch * iters_per_epoch

    if solver_max_iter // 10 < 200:
        solver_warmup_iters = solver_max_iter // 10
    else:
        solver_warmup_iters = 200

    solver_checkpoint_period = solver_max_iter // 5 + 1
    output_dir = "./results/ft_base/{}/base_test/{}".format(part_name, name)


    arg_str = " SELF_STR '{}' ".format(self_str) + \
              " OUTPUT_DIR '{}'".format(output_dir)

    cmd = base_cmd + arg_str

    print(cmd)
    os.system(cmd)

