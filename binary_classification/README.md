# Experiments on Binary classification

Please download the [dataset] (https://www.kaggle.com/datasets/kmader/rsna-bone-age) in "dataset" directoy.

## Training

To train the gender classification model, please run:

```
bash ./scripts/train.sh
```


## OOD Detection Evaluation

To reproduce our TAP-Mahalanobis and TAP-Ensemble results, please run the following commands for feature extraction followed by :
```
bash ./scripts/get_features.sh
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

## OOD Detection Results

### Results
OOD detection performance comparison between TAPUDD method and baselines. Our method detects samples from Textures more OOD compared to samples from iNaturalist, SUN, Places (similar to the way humans perceive).

![results](images/multi-class-results.png)

### Analysis
***(top)*** Examples of ID images sampled from Imagenet and OOD images sampled from iNaturalist, SUN, Places, and Textures dataset; ***(middle)*** Point-density based PCA visualization to demonstrate the location and density of ID and OOD datasets; ***(bottom)*** Point-density based PCA visualization of ID dataset overlapped by PCA of different OOD datasets to demonstrate the location and density of different OOD datasets relative to the ID dataset. Dataset images ***(top)*** and PCA ***(bottom)*** demonstrates that Textures is more different from Imagenet than other three OOD datasets.
![pca_results](images/pca5.jpeg)
