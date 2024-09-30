export CUDA_VISIBLE_DEVICES=0
python tools/calc_metrics_for_dataset.py \
--fake_data_path /131_data/jihwan/2023_fifo/latte-fifo/test_fifo_n4_ld-frames \
--mirror 1 --gpus 1 --resolution 256 \
--metrics isv2048_ucf  \
--verbose 0 --use_cache 0