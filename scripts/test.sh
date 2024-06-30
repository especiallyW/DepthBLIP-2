
#!/bin/bash
export PYTHONUNBUFFERED=1
# export HF_HOME=/path/to/config/bert-base-uncased/
# export TORCH_HOME=/path/to/config/blip2_pretrained_vitL/

python main.py \
--method second \
--dataset NYU \
--data_root_path ./datasets/NYU/ \
--test_file ./datasets/NYU/test/nyudepthv2_test_files_with_gt_dense.txt \
--class_name all \
--test_log_save \
--log_result_dir ./log_results \

