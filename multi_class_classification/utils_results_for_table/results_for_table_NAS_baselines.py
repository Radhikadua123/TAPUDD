import numpy as np
adjust_scale = list(np.arange(0,20,1)/10)
adjust_scale += list(np.arange(20,80,5)/10)

for method in ["mos"]:
    AUROC = []
    AUPR_in = []
    AUPR_out = []
    FPR_95 = []
    for var in adjust_scale:
        with open(f'/home/radhika/mos1/checkpoints/test_log/NAS_DETECTION_bright_test_BiT-S-R101x1_{method}/{var}/train.log') as f:
            lines = f.readlines()
        # AUROC.append(lines[-5].split("AUROC: ")[-1])
        # AUPR_in.append(lines[-4].split("AUPR (In): ")[-1])
        # AUPR_out.append(lines[-3].split("AUPR (Out): ")[-1])
        # FPR_95.append(lines[-2].split("FPR95: ")[-1])
        AUROC.append(lines[-4].split("AUROC using sk.auc: ")[-1])
        AUPR_in.append(lines[-3].split("AUPR (In) using sk.auc: ")[-1])
        AUPR_out.append(lines[-2].split("AUPR (Out) using sk.auc: ")[-1])
        # FPR_95.append(lines[-2].split("FPR95: ")[-1])

    print("-" * 50)
    print("METHOD: ", method)

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
