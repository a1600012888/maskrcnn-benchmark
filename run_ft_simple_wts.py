import os

part_name = "time-weather-scene-part"
NGPUS = 2
batch_size = 16 * NGPUS
num_epoch = 2
stpes = [2]

base_cmd = "python3 -m torch.distributed.launch --nproc_per_node={} --master_port=1235 tools/train_net.py".format(NGPUS) +\
           " --config-file './configs/template/ft_self_simple.yaml' "

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

    solver_warmup_iters = 0
    #if solver_max_iter // 10 < 100:
    #    solver_warmup_iters = solver_max_iter // 10
    #else:
    #    solver_warmup_iters = 100

    solver_checkpoint_period = solver_max_iter // 5
    output_dir = "./results/ft_base/{}/simple/{}".format(part_name, name)


    arg_str = " SELF_STR '{}' ".format(self_str) + \
              " OUTPUT_DIR '{}'".format(output_dir) + \
              " SOLVER.STEPS " + "'({},)'".format(*solver_steps) + \
              " SOLVER.MAX_ITER " + "{}".format(solver_max_iter) + \
              ' SOLVER.WARMUP_ITERS ' + "{}".format(solver_warmup_iters) + \
              ' SOLVER.CHECKPOINT_PERIOD ' + "{}".format(solver_checkpoint_period)

    cmd = base_cmd + arg_str

    print(cmd)
    os.system(cmd)

