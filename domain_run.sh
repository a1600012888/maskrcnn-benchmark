ulimit -n 4096
export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/bdd_faster_rcnn_fbnet_domain_2x.yaml"
