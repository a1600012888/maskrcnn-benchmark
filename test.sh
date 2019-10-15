ulimit -n 4096
export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=9999 tools/test_net.py --config-file "./configs/bdd_faster_rcnn_fbnet_lr2x_aug2_planb.yaml"
#CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "ty_tools/time_lr2x_lam0.2.yaml"
