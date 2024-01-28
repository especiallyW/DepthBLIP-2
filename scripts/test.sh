
#!/bin/bash
export PYTHONUNBUFFERED=1
export HF_HOME=/path/to/config/bert-base-uncased/
export TORCH_HOME=/path/to/config/blip2_pretrained_vitL/

python main.py \
--test_file ./datasets/NYU/nyudepthv2_test_files_with_gt_dense.txt \
--data_root_path ./datasets/NYU/test \
--log_save \
--log_result_dir ./log_result \
