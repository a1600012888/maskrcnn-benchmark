ulimit -n 4096
export NGPUS=4
export NNODES=2
export MASTER_ADDR='10.138.15.202'
python -m torch.distributed.launch --nproc_per_node=$NGPUS --nnodes=$NNODES --node_rank=0  --master_addr=$MASTER_ADDR --master_port=1234 tools/train_net.py --config-file "configs/embed_jitter/bdd_faster_rcnn_fbnetd_domain_time_1x_lr2x_aug.yaml"
