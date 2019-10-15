ulimit -n 4096
export NGPUS=4
#python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "./configs/bdd_faster_rcnn_fbnet_lr2x_aug2_planb.yaml"
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/ft_time_2_planb.yaml"
