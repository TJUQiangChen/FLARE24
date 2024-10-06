# 🔥 FLARE24 Solution

This repository is the official implementation of [A 3D Unsupervised Domain Adaption Framework Combining Style Translation and Self-Training for Abdominal Organs Segmentation](https://openreview.net/forum?id=oSbUYnIDs9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DMICCAI.org%2F2024%2FChallenge%2FFLARE%2FAuthors%23your-submissions)) of Team tju_vil_pioneers on FLARE24 challenge.

## 🔍 Overview
This work addresses the challenge of adapting CT-trained segmentation models to MR images by generating synthetic MR data, employing self-training strategies, and a two-stage segmentation framework in the FLARE24 dataset. For more details, see the pipeline diagram below:

<div align=center>
<img src="./assets/pipeline.png" alt="Pipeline" width="800"/>
</div>

## ⚙️ Environments and Requirements
* Ubuntu 20.04.6 LTS
* Intel(R) Xeon(R) Platinum 8153 CPU @ 2.00GHz, RAM 192GB , NVIDIA Tesla V100 (32G)
* CUDA >= 11.3
* python >= 3.7.13

To set up the environment, follow these steps:

```
conda create -n FLARE24 python=3.7.13
conda activate FLARE24
pip install -r requirements.txt
```
> PS:  should install torch==1.12.0+cu113 before pip install requirement. 
```
pip install torch==1.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
## 💾 Dataset
The training Data and validation data are provided by the [FLARE24](https://www.codabench.org/competitions/2296/). In short, there are 2200 partial labeled CT and ? unlabeled MR data for training, 100 public cases for validation and 200 hidden cases for the final test.

我们的数据目录结构如下图
TODO: 补充

## 🪄 Preprocessing

我们根据模型训练的阶段，包含了多个步骤的预处理。
第一阶段，我们仅仅使用CT数据和领域迁移的数据
For Step 1 Style Translation
```
python xxx
```
第二阶段，我们使用训练好的模型迭代生成Big SegNet的伪标签用来进行self-train
For Step 2 self-Training Strategy
```
python xxx
```
Now you can train your models.
第三阶段，我们使用第二阶段训练好的模型生成Small SegNet的伪标签，进行最终模型的训练
For Step 3 Two-Stage Segmentation
```
python xxxx
```
Now you can train your models.


# 🖥️ Train

Training our model consists of ? main steps, as depicted in the pipeline figure above. Follow these instructions to conduct the training process effectively:

### Step 1: 佳燨补充， 域迁移算法
Firstly, prepare a dataset of 250 images with 13-organ labels. Use this dataset to train a large organ model. 

```bash
nnUNetv2_train TASK_ID 3d_fullres all -tr STUNetTrainer_large
```

### Step 2: Generate Pseudo Labels for Step2 
With the model trained in Step 1, generate pseudo labels for 1494 tumor-annotated images.

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d TASK_ID -c 3d_fullres -f all -tr STUNetTrainer_large
```

### Step 3: Merge Labels and Train Large Model
Merge the pseudo organ labels and actual tumor labels in 1497 images, and then train a large model using this merged dataset.

```bash
nnUNetv2_train TASK_ID 3d_fullres all -tr STUNetTrainer_large
```

### Step 4: Generate Pseudo Labels for All Images
Using the model trained in Step 3, generate pseudo labels for all 4000 images in the dataset.

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d TASK_ID -c 3d_fullres -f all -tr STUNetTrainer_large
```

### Step 5: Train Small Model for Deployment
Finally, train a small model on the dataset of 4000 images. This model will be used for deployment.

```bash
nnUNetv2_train TASK_ID 3d_fullres all -tr STUNetTrainer_base
```

You can download trained models here:

TODO:补充link trained on the above dataset with the above code.

2. To fine-tune the model on a customized dataset, run this command:

```bash
nnUNetv2_train TASK_ID 3d_fullres all -tr STUNetTrainer_base
```

## 🗳️ Inference

1. To infer the testing cases, run this command:

```bash
nnUNetv2_train TASK_ID 3d_fullres all -tr STUNetTrainer_base
```
2. Docker containers on [docker container_link](354)

Will be released soon.

## 📊 Evaluation

To compute the evaluation metrics, run:

```eval
python eval.py --seg_data <path_to_inference_results> --gt_data <path_to_ground_truth>
```

>Describe how to evaluate the inference results and obtain the reported results in the paper.



## 📋 Results

Our method achieves the following performance on [FLARE24](https://www.codabench.org/competitions/2296/)

| Dataset Name       | DSC(%) | NSD(%) |
|--------------------|:------:|:------:|
| Validation Dataset | 79.42% | 86.46% |
| Test Dataset       |   ?    |   ？    |




## 💕 项目成员 <a id="项目成员"></a>
#TODO
- [xx](xx) 

## 🖊️ Citation <a id="Citation"></a>

```bibtex
todo
```

## Contributing

## 🉑开源许可证 <a id="开源许可证"></a>

该项目采用 [Apache License 2.0 开源许可证](LICENSE) 同时，请遵守所使用的模型与数据集的许可证。
## Acknowledgement

 We thank the contributors of [public FLARE24 datasets](?).

> We thank the contributors of public datasets. 