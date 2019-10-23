ulimit -n 4096
export NGPUS=8
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=9999 tools/test_net_pool.py --config-file "./configs/template/model_pool_test.yaml"
