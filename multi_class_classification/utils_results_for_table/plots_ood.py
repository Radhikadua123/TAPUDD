import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12})

datasets = ["dtd", "iNaturalist", "Places", "SUN"]

colors = [ "plum", "tan", "coral", "cornflowerblue", "cyan"]

# colors = sns.color_palette("muted", 4)
using_AUC = True
clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32]

i = 0
for dataset in datasets:
    AUROC = []
    AUPR_in = []
    AUPR_out = []
    FPR_95 = []
    for cluster in clusters:
        with open(f'/home/radhika/mos1/checkpoints/test_log/test_BiT-S-R101x1_GMM_FULL_Mahala_{dataset}/num_samples1281167/cluster{cluster}/train.log') as f:
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
            FPR_95.append(100*float(lines[-4].split("FPR95: ")[-1]))

    if(dataset == "dtd"):
        label = "Textures"
    else:
        label = dataset
    
    # sns.lineplot(x=clusters, y=FPR_95, color=colors[i], label=label)
    sns.lineplot(x=clusters, y=AUROC, color=colors[i], label=label)

    i += 1
plt.xlabel('Number of clusters',fontsize=15)
# plt.ylabel('FPR95')
plt.ylim([50, 100])
plt.ylabel('AUROC',fontsize=15)
plt.title("Multi-class Classification",fontsize=15)
# plt.legend(title = "Dataset", fancybox=True).get_frame().set_alpha(0.7)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fancybox=True, shadow=False, ncol=2,fontsize=15)
# plt.legend(loc=10, fancybox=True).get_frame().set_alpha(0.7)
# plt.savefig(f'/home/radhika/assign/figures_for_paper/mos_ood/plot_fpr95_umahala_cluster_ablation.png', bbox_inches='tight', dpi=300)
plt.savefig(f'/home/radhika/assign/figures_for_paper/mos_ood/plot_auroc_umahala_cluster_ablation.png', bbox_inches='tight', dpi=300)
plt.close()

  
colors = sns.color_palette("Blues", 5)

i = 0
ensemble_vars = [ "average", "trimmed_average", "seesaw", "top", "bottom"]
ensemble_vars_for_labels = ["Average", "Trimmed Average", "Seesaw", "Top", "Bottom"]
for ensemble in ensemble_vars:
    AUROC = []
    AUPR_in = []
    AUPR_out = []
    FPR_95 = []
    for dataset in datasets:
        with open(f'/home/radhika/mos1/checkpoints/test_log/test_BiT-S-R101x1_GMM_FULL_Mahala_Ensemble/{ensemble}/{dataset}/num_samples1281167/train.log') as f:
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
            FPR_95.append(100*float(lines[-4].split("FPR95: ")[-1]))

    label = ensemble_vars_for_labels[i]
    
    r1 = np.arange(4)
    r2 = [1+i*0.3, 3+i*0.3, 5+i*0.3, 7+i*0.3] 

    plt.bar(r2, AUROC, width = 0.3, color = colors[i], capsize=7, label=label)

    i += 1

plt.xticks([1+0.5, 3+0.5, 5+0.5, 7+0.5], ["Textures", "iNaturalist", "Places", "SUN"],fontsize=15)

# plt.ylabel('FPR95')
plt.ylabel('AUROC',fontsize=15)
plt.title("Multi-class Classification",fontsize=15)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=False, ncol=3)
plt.ylim([50, 100])
plt.savefig(f'/home/radhika/assign/figures_for_paper/mos_ood/plot_auroc_tauood_ensemble_ablation.png', bbox_inches='tight', dpi=300)
plt.close()
