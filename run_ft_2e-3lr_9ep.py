import os

part_name = "time-weather-part"
NGPUS = 4
batch_size = 16 * NGPUS
num_epoch = 9
stpes = [6, 8]

base_cmd = "python3 -m torch.distributed.launch --nproc_per_node={} --master_port=9999 tools/train_net.py".format(NGPUS) +\
           " --config-file './configs/template/ft_self_base_2e-3lr.yaml' "

img_dir = os.path.join("./datasets/bdd100k/data_part",
                       part_name, "images/train")

val_img_dir = os.path.join("./datasets/bdd100k/data_part",
                       part_name, "images/val")

all_names = os.listdir(img_dir)
val_names = os.listdir(val_img_dir)
all_names = set(all_names).intersection(set(val_names))
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

    if solver_max_iter // 10 < 100:
        solver_warmup_iters = solver_max_iter // 10
    else:
        solver_warmup_iters = 100

    solver_checkpoint_period = solver_max_iter // 5
    output_dir = "./results/ft_base/{}/9ep_2e-3lr/{}".format(part_name, name)


    arg_str = " SELF_STR '{}' ".format(self_str) + \
              " OUTPUT_DIR '{}'".format(output_dir) + \
              " SOLVER.STEPS " + "'({}, {})'".format(*solver_steps) + \
              " SOLVER.MAX_ITER " + "{}".format(solver_max_iter) + \
              ' SOLVER.WARMUP_ITERS ' + "{}".format(solver_warmup_iters) + \
              ' SOLVER.CHECKPOINT_PERIOD ' + "{}".format(solver_checkpoint_period)

    cmd = base_cmd + arg_str

    print(cmd)
    os.system(cmd)

