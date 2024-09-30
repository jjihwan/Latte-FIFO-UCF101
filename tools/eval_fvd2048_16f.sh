export CUDA_VISIBLE_DEVICES=0
python tools/calc_metrics_for_dataset.py \
--real_data_path /data/jihwan/UCF-101-frames \
--fake_data_path /131_data/jihwan/2023_fifo/latte-fifo/test_fifo_n4_ld-frames \
--mirror 1 --gpus 1 --resolution 256 \
--metrics fvd2048_16f  \
--verbose 0 --use_cache 0