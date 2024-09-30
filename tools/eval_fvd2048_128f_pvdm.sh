export CUDA_VISIBLE_DEVICES=0,1,2,3
python tools/calc_metrics_for_dataset.py \
--real_data_path /131_data/jihwan/data/UCF-101-test-frames \
--fake_data_path /131_data/jihwan/2023_fifo/latte-fifo/test_pvdm-frames \
--mirror 1 --gpus 4 --resolution 256 \
--metrics fvd2048_128f  \
--verbose 0 --use_cache 0