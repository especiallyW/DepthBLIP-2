## 语言

[English](README.md) | **简体中文**

## 文件目录结构

```
output
|–– configs/    # 模型加载的配置文件。
|   |–– bert-base-uncased/
|   |–– blip2_pretrained_vitL/
|–– datasets/   # 数据集存放与加载脚本。
|–– models/     # DepthBLIP-2模型。
|–– scripts/    # 快速运行项目的shell脚本。
|–– utils/      # 公共使用脚本。
|–– main.py    # 项目运行接口。
```

## 运行

### 所有场景

你可以选择为`train.py`配置参数进行运行，也可以使用我们提供的`test.sh`脚本进行运用。

若使用`test.sh`进行运行，你需要处于当前项目的根目录，接着运行 `bash scripts/test.sh`。同时，请注意配置`HF_HOME`和`TORCH_HOME`的环境变量，它们分别对应hugging face和lavis中模型加载的环境变量。

在完成实验后，你能在`/path/to/project/root/log_result/result.txt`中发现评估结果。其大致形式如下：

```
---->class_name: all
---->depth_templates: ['This {} is {}']
---->obj_classes: ['object']
---->depth_classes: ['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far', 'unseen']
---->bin_list: [1.0, 1.75, 2.25, 2.5, 2.75, 3.0, 3.5]
* * Avg abs_diff : 0.934, a1 : 0.384, a2 : 0.683, a3 : 0.856, abs_rel : 0.364, log10 : 0.156, rmse : 1.172
```

### 指定场景

若需要测试指定场景的性能，请务必添加形参变量`--class_name scene_name`以此指定场景。

## 说明

- 若想更改数据集路径、批量大小等配置参数，请参考`main.py`中的`parser`进行相应修改。
- 我们的代码基于[DepthCLIP](https://github.com/Adonis-galaxy/DepthCLIP)进行修改，所以会有部分地方与该项目类似。

## 致谢

我们的代码借助很多：

- [LAVIS](https://github.com/salesforce/LAVIS)
