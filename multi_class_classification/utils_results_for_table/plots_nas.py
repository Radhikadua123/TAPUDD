import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 10})

adjust_scale = list(np.arange(0,20,1)/10)
adjust_scale += list(np.arange(20,80,5)/10)

ensemble = "bottom"
clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32]
using_AUC = True
random_state=0

# colors = ['r', 'b', 'y', 'c','m', 'g']
# colors = [ "tan", "khaki", "mediumpurple", "mediumturquoise", "coral", 'r', 'b', 'm', 'g', 'y', 'c', "cyan", "cyan"]
colors = sns.color_palette("Spectral", 13)
# colors = sns.color_palette("cubehelix", 13)

print(colors)
temp_auroc = []
temp_scale = []
temp_auroc2 = []
        
AUROC = []
AUPR_in = []
AUPR_out = []
FPR_95 = []
adjust_scale = list(np.arange(0,20,1)/10)
adjust_scale += list(np.arange(20,80,5)/10)
for var in adjust_scale:
    with open(f'/home/radhika/mos1/checkpoints/test_log/NAS_DETECTION_bright_test_BiT-S-R101x1_GMM_FULL_Mahala_Ensemble/{ensemble}/{var}/num_samples1281167/train.log') as f:
    # with open(f'/home/radhika/assign/binary_classification/results_ce/logs/ood_scores/NAS_DETECTION_bright_test_GMM_FULL_Mahala/seed_{input_seed}/gmm_random_state_{random_state}/cluster{cluster}/{var}/ood_scores.log') as f:
        lines = f.readlines()
    
    if(using_AUC != True):
        AUROC.append(lines[-7].split("AUROC: ")[-1])
        AUPR_in.append(lines[-6].split("AUPR (In): ")[-1])
        AUPR_out.append(lines[-5].split("AUPR (Out): ")[-1])
        FPR_95.append(lines[-4].split("FPR95: ")[-1])
    else:
        AUROC.append(100*float(lines[-3].split("AUROC using sk.auc: ")[-1]))
        AUPR_in.append(lines[-2].split("AUPR (In) using sk.auc: ")[-1])
        AUPR_out.append(lines[-1].split("AUPR (Out) using sk.auc: ")[-1])
        FPR_95.append(lines[-4].split("FPR95: ")[-1])



print(temp_auroc, temp_scale)
# sns.lineplot(x=adjust_scale, y=AUROC, color=colors[0], label=f"TAU-OOD: Bottom")


i=1
for cluster in clusters:
    temp_auroc = []
    temp_scale = []
    temp_auroc2 = []
        
    AUROC = []
    AUPR_in = []
    AUPR_out = []
    FPR_95 = []
    adjust_scale = list(np.arange(0,20,1)/10)
    adjust_scale += list(np.arange(20,80,5)/10)
    for var in adjust_scale:
        # with open(f'/home/radhika/assign/binary_classification/results_ce/logs/ood_scores/NAS_DETECTION_bright_test_GMM_FULL_Mahala_Ensemble/{ensemble}/seed_{input_seed}/gmm_random_state_{random_state}/{var}/ood_scores.log') as f:
        with open(f'/home/radhika/mos1/checkpoints/test_log/NAS_DETECTION_bright_test_BiT-S-R101x1_GMM_FULL_Mahala/{var}/num_samples1281167/cluster{cluster}/train.log') as f:
            lines = f.readlines()
        
        if(using_AUC != True):
            AUROC.append(lines[-7].split("AUROC: ")[-1])
            AUPR_in.append(lines[-6].split("AUPR (In): ")[-1])
            AUPR_out.append(lines[-5].split("AUPR (Out): ")[-1])
            FPR_95.append(lines[-4].split("FPR95: ")[-1])
        else:
            AUROC.append(100*float(lines[-3].split("AUROC using sk.auc: ")[-1]))
            AUPR_in.append(lines[-2].split("AUPR (In) using sk.auc: ")[-1])
            AUPR_out.append(lines[-1].split("AUPR (Out) using sk.auc: ")[-1])
            FPR_95.append(lines[-4].split("FPR95: ")[-1])

    
    # sns.lineplot(x=temp_scale, y=temp_auroc, color=colors[i], label=f"Cluster {cluster}", linestyle='--')
    sns.lineplot(x=adjust_scale, y=AUROC, color=colors[i], label=f"Cluster {cluster}", linestyle='--')
    
    i += 1

    print("-" * 50)
plt.xlabel('Brightness')
plt.ylabel('AUROC')
# plt.legend(loc=10, fancybox=True).get_frame().set_alpha(0.7)
plt.savefig(f'/home/radhika/assign/figures_for_paper/mos_nas/plot_auroc_umahala_cluster_ablation.png', bbox_inches='tight', dpi=300)
plt.close()