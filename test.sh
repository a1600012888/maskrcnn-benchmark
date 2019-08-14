export NGPUS=4
CUDA_VISIBLE_DEVICES=10,11,12,13 python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/test_net.py --config-file "mm-configs/bdd_faster_rcnn_fbnet_domain_time_1x_lr2x.yaml"
