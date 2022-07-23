# Experiments on Multi-class classification

We use the datasets and code from [this](https://github.com/deeplearning-wisc/large_scale_ood) repository for training the model for classification task and for the baseline OOD detection methods. Please refer to the aforementioned repository for detailed information on dataset and pretrained model preparation.

## Usage

### 1. Dataset Preparation

#### In-distribution (ID) dataset
Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in some directory. We stored the train data in `/home/data_storage/imagenet/v12/train` and val data in `/home/data_storage/imagenet/v12/val`

#### Out-of-distribution (OOD) dataset
Please download the OOD datasets following the instructions given in [this](https://github.com/deeplearning-wisc/large_scale_ood) repository and put it into some directory. We stored the OOD datasets in `/home/data_storage/ood_datasets/data/ood_data/`.

### 2. Pre-trained Model Preparation

Please download the [BiT-S pre-trained model families](https://github.com/google-research/big_transfer)
and put them into the folder `./bit_pretrained_models`.


### 3. Group-softmax/Flat-softmax Model Finetuning

For flat-softmax finetuning (used for TAPUDD, TAP-Mahalanobis and baselines), please modify `/home/data_storage/imagenet/v12` with ID dataset path and run:

```
bash ./scripts/finetune_flat_softmax.sh /home/data_storage/imagenet/v12
```

For group-softmax finetuning (MOS), please modify `/home/data_storage/imagenet/v12` with ID dataset path and run:

```
bash ./scripts/finetune_group_softmax.sh /home/data_storage/imagenet/v12
```


### 3. OOD Detection Evaluation

To reproduce our TAP-Mahalanobis and TAP-Ensemble results, please run the following commands for feature extraction followed by OOD Detection. Please modify `/home/data_storage/imagenet/v12` with ID dataset path and `/home/data_storage/radhika/data/ood_data` with OOD dataset path.
```
bash ./scripts/get_features.sh iNaturalist(/SUN/Places/Textures) /home/data_storage/imagenet/v12 /home/data_storage/radhika/data/ood_data
bash ./scripts/test_ours.sh 
bash ./scripts/test_ours_ensemble.sh 
```

To reproduce baseline approaches (MSP, ODIN, Energy, Mahalanobis, KL_Div), please run:
```
bash ./scripts/test_baselines.sh MSP(/ODIN/Energy/Mahalanobis/KL_Div) 
```

To reproduce MOS, please run:
```
bash ./scripts/test_mos.sh
```

Note: before testing Mahalanobis, make sure you have tuned and saved its hyperparameters first by running:
```
bash ./scripts/tune_mahalanobis.sh
```

### OOD Detection Results

### Results
OOD detection performance comparison between TAPUDD method and baselines. Our method detects samples from Textures more OOD compared to samples from iNaturalist, SUN, Places (similar to the way humans perceive).

![results](images/multi-class-results.png)

### Analysis
***(top)*** Examples of ID images sampled from Imagenet and OOD images sampled from iNaturalist, SUN, Places, and Textures dataset; ***(middle)*** Point-density based PCA visualization to demonstrate the location and density of ID and OOD datasets; ***(bottom)*** Point-density based PCA visualization of ID dataset overlapped by PCA of different OOD datasets to demonstrate the location and density of different OOD datasets relative to the ID dataset. Dataset images ***(top)*** and PCA ***(bottom)*** demonstrates that Textures is more different from Imagenet than other three OOD datasets.
![pca_results](images/pca5.jpeg)
