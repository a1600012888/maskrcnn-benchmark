import os

import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=20)
    parser.add_argument('--c', type=str, default="./configs/new/baseline_e4_aug1_new.yaml")

    args = parser.parse_args()

    NGPUs = 4
    print(os.listdir('./'))
    run_cmd = 'python3 -m torch.distributed.launch --nproc_per_node={} '.format(NGPUs) + \
              'tools/train_net.py --config-file {}'.format(args.c)

    source_dir = './datasets/bdd100k/copy.domain_embedding_val'
    target_dir = './datasets/bdd100k/domain_embedding_val_new'
    result_dir = './results/new/4e_weather_time_aug1/'

    shuflle_cmd = 'python3 ty_tools/shuflle_domain_embed.py --p={} --t={}'.format(source_dir, target_dir)
    parse_cmd = 'python3 ty_tools/parse_shuffle_result.py --p={}'.format(result_dir)
    for i in range(args.num):
        os.system(shuflle_cmd)
        os.system(run_cmd)
        os.system(parse_cmd)
