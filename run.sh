ulimit -n 2048
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/bdd_faster_rcnn_fbnet.yaml" 
