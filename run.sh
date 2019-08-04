ulimit -n 4096
export NGPUS=16
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/bdd_faster_rcnn_fbnet_2x.yaml" 
