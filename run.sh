ulimit -n 4096
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/embed_jitter/bdd_faster_rcnn_fbnetd_domain_time_1x_lr2x.yaml"
