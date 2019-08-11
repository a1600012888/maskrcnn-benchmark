ulimit -n 4096
export NGPUS=4
export NNODES=2
export MASTER_ADDR='10.138.15.192'
python -m torch.distributed.launch --nproc_per_node=$NGPUS --nnodes=$NNODES --node_rank=1  --master_addr=$MASTER_ADDR --master_port=1234 tools/train_net.py --config-file "mm-configs/bdd_faster_rcnn_fbnet_domain_time_1x.yaml"
