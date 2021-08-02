import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import auc, roc_curve

from utils import get_date

# Set desired model names
model_list = ["110320_image-only_aug_ls0.1_TTA-5_seed3",
              "110320_shallow-only_ls0.1_seed4",
              "110420_probability-fusion_aug_ls0.1_TTA-5_seed4",
              "110420_feature-fusion_aug_ls0.1_TTA-5_seed3",
              "110420_hidden-feature-fusion-concat_aug_ls0.1_TTA-5_seed4"]

# Gather ROC curves for each model
roc_dict = {"ImageOnly": {}, "ShallowOnly": {}, "ProbabilityFusion": {}, "FeatureFusion": {}, "HiddenFeatureFusion": {}}
for i, model_name in enumerate(model_list):
    # Get predictions
    pred_df = pd.read_csv(os.path.join("results", model_name, "preds.csv"))
    y_test, y_prob = pred_df["y_true"].to_numpy(), pred_df["y_prob"].to_numpy()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auroc = auc(fpr, tpr)

    roc_dict[list(roc_dict.keys())[i]]["fpr"] = fpr
    roc_dict[list(roc_dict.keys())[i]]["tpr"] = tpr
    roc_dict[list(roc_dict.keys())[i]]["auc"] = auroc

# Plot ROC curves
roc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(roc_dict["ShallowOnly"]["fpr"], roc_dict["ShallowOnly"]["tpr"], lw=2,
        label=f"Non-Image-Only (AUC: {round(roc_dict['ShallowOnly']['auc'], 3)})")
ax.plot(roc_dict["ImageOnly"]["fpr"], roc_dict["ImageOnly"]["tpr"], lw=2,
        label=f"Image-Only (AUC: {round(roc_dict['ImageOnly']['auc'], 3)})")
ax.plot(roc_dict["FeatureFusion"]["fpr"], roc_dict["FeatureFusion"]["tpr"], lw=2,
        label=f"Feature Fusion (AUC: {round(roc_dict['FeatureFusion']['auc'], 3)})")
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([-0.05, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('1 - Specificity', fontsize=13)
ax.set_ylabel('Sensitivity', fontsize=13)
ax.legend(loc="lower right", fontsize=11)

# Plot full ROC curve (comparing all models)
full_roc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(roc_dict["ShallowOnly"]["fpr"], roc_dict["ShallowOnly"]["tpr"], lw=2,
        label=f"Non-Image-Only (AUC: {round(roc_dict['ShallowOnly']['auc'], 3)})")
ax.plot(roc_dict["ImageOnly"]["fpr"], roc_dict["ImageOnly"]["tpr"], lw=2,
        label=f"Image-Only (AUC: {round(roc_dict['ImageOnly']['auc'], 3)})")
ax.plot(roc_dict["ProbabilityFusion"]["fpr"], roc_dict["ProbabilityFusion"]["tpr"], lw=2, linestyle=":",
        label=f"Probability Fusion (AUC: {round(roc_dict['ProbabilityFusion']['auc'], 3)})")
ax.plot(roc_dict["FeatureFusion"]["fpr"], roc_dict["FeatureFusion"]["tpr"], lw=2, linestyle="--",
        label=f"Feature Fusion (AUC: {round(roc_dict['FeatureFusion']['auc'], 3)})")
ax.plot(roc_dict["HiddenFeatureFusion"]["fpr"], roc_dict["HiddenFeatureFusion"]["tpr"], lw=2,
        label=f"Learned Feature Fusion (AUC: {round(roc_dict['HiddenFeatureFusion']['auc'], 3)})")
ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
ax.set_xlim([-0.05, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('1 - Specificity')
ax.set_ylabel('Sensitivity')
ax.legend(loc="lower right")

# Save figures
roc_plot.savefig(f"{get_date()}_roc.pdf", bbox_inches="tight")
full_roc_plot.savefig(f"{get_date()}_full_roc.pdf", bbox_inches="tight")