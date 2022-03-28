datasets = ["dtd", "iNaturalist", "Places", "SUN"]

using_AUC = True

for dataset in datasets:
    AUROC = []
    AUPR_in = []
    AUPR_out = []
    FPR_95 = []
    for method in ["seesaw", "top", "bottom", "average", "trimmed_average", "seesaw_scaling", "top_scaling", "bottom_scaling", "average_scaling", "trimmed_average_scaling"]:
        with open(f'/home/radhika/mos1/checkpoints/test_log/test_BiT-S-R101x1_GMM_FULL_Mahala_Ensemble/{method}/{dataset}/num_samples1281167/train.log') as f:
            lines = f.readlines()
        if(using_AUC != True):
            AUROC.append(lines[-7].split("AUROC: ")[-1])
            AUPR_in.append(lines[-6].split("AUPR (In): ")[-1])
            AUPR_out.append(lines[-5].split("AUPR (Out): ")[-1])
            FPR_95.append(lines[-4].split("FPR95: ")[-1])
        else:
            AUROC.append(lines[-3].split("AUROC using sk.auc: ")[-1])
            AUPR_in.append(lines[-2].split("AUPR (In) using sk.auc: ")[-1])
            AUPR_out.append(lines[-1].split("AUPR (Out) using sk.auc: ")[-1])
            FPR_95.append(lines[-4].split("FPR95: ")[-1])

    print("-" * 50)
    print("Dataset: ", dataset)

    print("### AUROC ###")
    for i in AUROC:
        print(i, end = " ")

    print("### AUPR_in ###")
    for i in AUPR_in:
        print(i, end = " ")

    print("### AUPR_out ###")
    for i in AUPR_out:
        print(i, end = " ")

    print("### FPR_95 ###")
    for i in FPR_95:
        print(i, end = " ")

    print("-" * 50)
