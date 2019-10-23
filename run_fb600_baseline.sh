ulimit -n 4096
export NGPUS=8
#python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "./configs/bdd_faster_rcnn_fbnet_lr2x_aug2_planb.yaml"
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/new/embed_jitter_time_weather_e4_aug1.yaml"
