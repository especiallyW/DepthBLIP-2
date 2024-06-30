## Language

**English** | [简体中文](README_CN.md)

## Directory Structure

```
output
|–– configs/    # Configuration files for model loading.
|   |–– bert-base-uncased/
|   |–– blip2_pretrained_vitL/
|   |–– blip2-opt-2.7b/
|   |–– blip2-vicuna-instruct-7b/
|-- blip2_extractor/ # py file of pre-training model.
|–– datasets/   # Scripts for dataset storage and loading.
|–– models/     # DepthBLIP-2 model.
|–– log_results/     # train/test's logging file.
|–– checkpoints/     # pth file after training model.
|–– scripts/    # Shell scripts for quick project execution.
|–– utils/      # Commonly used scripts.
|–– main.py    # Project execution interface.
```

## Running

### All Scenarios

You have the option to configure parameters for `train.py` and run it, or you can use our provided `test.sh` script.

If you choose to run `test.sh`, you need to be in the root directory of the current project, then
execute `bash scripts/test.sh`. Also, please make sure to set the `HF_HOME` and `TORCH_HOME` environment variables,
which correspond to the model loading environment variables in Hugging Face and Lavis, respectively.

Upon completion of the experiment, you can find the evaluation results
in `/path/to/project/root/log_results/method_name/test/N/test_result.txt`. The results are formatted as follows:

```
---->class_name: all
---->depth_templates: ['This {} is {}']
---->obj_classes: ['object']
---->depth_classes: ['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far', 'unseen']
---->bin_list: [1.0, 1.75, 2.25, 2.5, 2.75, 3.0, 3.5]
* * Avg abs_diff : 0.920, a1 : 0.393, a2 : 0.694, a3 : 0.861, abs_rel : 0.363, log10 : 0.153, rmse : 1.152
```

### Specific Scenarios

If you need to test the performance of a specific scenario, be sure to add the parameter `--class_name scenes_name` to
specify the scenario.

## Notes

- If you want to change configuration parameters such as the dataset path, batch size, training model etc., please refer to
  the `parser` in `main.py` for the corresponding modifications.
- Our code is modified based on [DepthCLIP](https://github.com/Adonis-galaxy/DepthCLIP), so there will be some
  similarities with that project.

## Acknowledgement

Our code borrows a lot from:

- [LAVIS](https://github.com/salesforce/LAVIS)

