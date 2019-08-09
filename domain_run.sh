ulimit -n 4096
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/bdd_faster_rcnn_fbnet_domain_scenes_1x.yaml"
