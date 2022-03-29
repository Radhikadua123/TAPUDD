# Experiments on Binary classification

Please download the [dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age) in "dataset" directoy.

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
NAS detection performance in binary classification task (gender prediction)for NAS shift of brightness in RSNA boneage dataset measured by AUROC. Highlighted row presents the performance on in-distribution dataset. MB and TAP-MB refers to Mahalanobis and TAP-Mahalanobis, respectively. Our proposed approaches, TAPUDD and TAP-Mahalanobis are more sensitive to NAS samples compared to competitive baselines.

![results](images/binary-class-results.png)
