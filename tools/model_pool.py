import os

base_result_dir = '/data/zhangtianyuan/maskrcnn-benchmark/results/ft_base'

data_name = "time-weather-part"

alg_name = 'bn_post'

base_model_dir = '/data/zhangtianyuan/models/bdd_fb_aug1_lr2.model'

val_data_dir = os.path.join('/data/zhangtianyuan/maskrcnn-benchmark/datasets/bdd100k/data_part',
                            data_name, 'images', 'val')

all_domain_names = os.listdir(val_data_dir)

model_root_dir = os.path.join(base_result_dir, data_name, alg_name)

all_model_names = os.listdir(model_root_dir)
#all_model_names = [a if os.path.exists(os.path.join(model_root_dir, a, 'model_final.pth')) else None for a in all_model_names]
t = []
for a in all_model_names:
    if os.path.exists(os.path.join(model_root_dir, a, 'model_final.pth')):
        t.append(a)
all_model_names = t

dusk = [str(i*4) for i in range(7)]
#all_model_names = list(set(all_model_names) - set(dusk))


all_model_path = [os.path.join(model_root_dir, model_name, 'model_final.pth') for model_name in all_model_names] + \
                 [base_model_dir]

id2model_path = {}
for k in all_domain_names:
    if k in all_model_names:
        id2model_path[int(k)] = os.path.join(model_root_dir, k, 'model_final.pth')
    else:
        id2model_path[int(k)] = base_model_dir
