import numpy as np
adjust_type="bright"
cluster = 2
if(adjust_type=="bright"):
    adjust_scale = list(np.arange(0,20,2)/10)
    adjust_scale += list(np.arange(20,70,5)/10)
else:
    adjust_scale = list(range(1,21))

using_AUC = True

mean_AUROC = []
mean_AUPR_in = []
mean_AUPR_out = []
mean_FPR_95 = []
for seed in [0, 1, 2, 3, 4, 663, 1458, 1708, 1955, 7130]:
    AUROC = []
    AUPR_in = []
    AUPR_out = []
    FPR_95 = []
    for var in adjust_scale:
        with open(f'results/logs_ood_scores/NAS_DETECTION_{adjust_type}_test_TAP_Mahalanobis/seed_{seed}/cluster{cluster}/{var}/ood_scores.log') as f:
            lines = f.readlines()
        
        AUROC.append(float(lines[-3].split("AUROC using sk.auc: ")[-1]))
        AUPR_in.append(float(lines[-2].split("AUPR (In) using sk.auc: ")[-1]))
        AUPR_out.append(float(lines[-1].split("AUPR (Out) using sk.auc: ")[-1]))
        FPR_95.append(float(lines[-4].split("FPR95: ")[-1]))

    mean_AUROC.append(AUROC)
    mean_AUPR_in.append(AUPR_in)
    mean_AUPR_out.append(AUPR_out)
    mean_FPR_95.append(FPR_95)

averaged_AUROC = np.mean(np.array(mean_AUROC), axis = 0)
averaged_AUPR_in = np.mean(np.array(mean_AUPR_in), axis = 0)
averaged_AUPR_out = np.mean(np.array(mean_AUPR_out), axis = 0)
averaged_FPR_95 = np.mean(np.array(mean_FPR_95), axis = 0)

std_AUROC = np.std(np.array(mean_AUROC), axis = 0)
std_AUPR_in = np.std(np.array(mean_AUPR_in), axis = 0)
std_AUPR_out = np.std(np.array(mean_AUPR_out), axis = 0)
std_FPR_95 = np.std(np.array(mean_FPR_95), axis = 0)

print("-" * 50)
print("METHOD: TAP-Mahalanobis")

print("### MEAN AUROC ###")
for i in averaged_AUROC:
    print(round(i*100,1))

print("### MEAN AUPR_in ###")
for i in averaged_AUPR_in:
    print(round(i*100,1))

print("### MEAN AUPR_out ###")
for i in averaged_AUPR_out:
    print(round(i*100,1))

print("### MEAN FPR_95 ###")
for i in averaged_FPR_95:
    print(round(i*100,1))

print("### STD AUROC ###")
for i in std_AUROC:
    print(round(i*100,1))

print("### STD AUPR_in ###")
for i in std_AUPR_in:
    print(round(i*100,1))

print("### STD AUPR_out ###")
for i in std_AUPR_out:
    print(round(i*100,1))

print("### STD FPR_95 ###")
for i in std_FPR_95:
    print(round(i*100,1))


print("-" * 50)