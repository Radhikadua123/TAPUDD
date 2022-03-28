# TAPUDD: Task Agnostic and Post-hoc </br> Unseen Distribution Detection
This is the *Pytorch Implementation* for the paper TAPUDD: Task Agnostic and Post-hoc Unseen Distribution Detection

**Authors**: [Radhika Dua](https://radhikadua123.github.io), [Seongjun Yang](https://jaehong31.github.io), [Yixuan Sharon Li](https://pages.cs.wisc.edu/~sharonli/), [Edward Choi](https://mp2893.com)

## Abstract
Despite the recent advances in out-of-distribution(OOD) detection, anomaly detection, and uncertainty estimation tasks, there do not exist a task-agnostic and post-hoc approach.
To address this limitation, we design a novel clustering-based ensembling method, called **T**ask **A**gnostic and **P**ost-hoc **U**nseen **D**istribution **D**etection (TAPUDD) that utilizes the features extracted from the model trained on a specific task. Explicitly, it comprises of *TAP-Mahalanobis*, which
clusters the training datasets' features and determines the minimum Mahalanobis distance of the test sample from all clusters. Further, we propose the *Ensembling module*
that aggregates the computation of iterative TAP-Mahalanobis for a different number of clusters 
to provide reliable and efficient cluster computation.
Through extensive experiments on synthetic and real-world datasets, we observe that our approach can detect unseen samples effectively across diverse tasks and performs better or on-par with the existing baselines. To this end, we eliminate the necessity of determining the optimal value of the number of clusters and demonstrate that our method is more viable for large-scale classification tasks.

__Contribution of this work__
- We propose a novel task-agnostic and post-hoc approach, **TAPUDD**, to detect unseen samples across diverse tasks like classification, regression, etc. 
- For the first time, we empirically show that a single approach can be used for multiple tasks with stable performance. We conduct exhaustive experiments on synthetic and real-world datasets for regression, binary classification, and multi-class classification tasks to demonstrate the effectiveness of our method.
- We also conduct ablation studies to illustrate the effect of using a different number of clusters in *TAP-Mahalanobis* and different ensembling strategies in TAPUDD. We observe that TAPUDD with different ensembling strategies performs better or on-par with *TAP-Mahalanobis*.


## Prerequisites
```
$ conda env create -f environment.yml
```

## Experiments
- Synthetic datasets
- Binary classification
- Multi-class classification
- Regression

## Contributing
We'd love to accept your contributions to this project. Please feel free to open an issue, or submit a pull request as necessary. If you have implementations of this repository in other ML frameworks, please reach out so we may highlight them here.
