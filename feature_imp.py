import os
import time
import sys
from copy import deepcopy

import tqdm
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import io

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FeatureImpDataset
from utils import get_date, to_one_hot, predict_prob
from models import ShallowFFNN, FeatureFusion, LearnedFeatureFusion, ProbabilityFusion

def permutation_importance(pt_model, data, data_loader, feature_mask, device, fusion, iters=30):
    # Get original predicted probabilities on test set
    y_prob = predict_prob(pt_model, data_loader, device, fusion=fusion)

    # Get original (base) AUC on test set
    base_err = roc_auc_score(data.y_test, y_prob)
    
    np.random.seed(0)
    importances = []
    for idx in feature_mask:
        imp_vals = []
        for _ in range(iters):
            if isinstance(idx, list):
                new_X = deepcopy(data.orig_meta_test)

                # Collapse dummy variables into categorical (n,)
                cat_X = np.argmax(new_X[:, idx], axis=1)
                cat_X_perm = np.random.permutation(cat_X)  # permute

                # Convert back to one-hot/dummy
                new_X[:, idx] = to_one_hot(cat_X_perm)  
            else:
                # Permute across dim 0
                new_X = deepcopy(data.orig_meta_test)
                new_X[:, idx] = np.random.permutation(data.orig_meta_test[:, idx])

            # Change underlying meta_test to permuted version
            data.meta_test = new_X

            # Get predicted probabilities on permuted test set
            y_prob = predict_prob(pt_model, data_loader, device, fusion=fusion)

            # Recalculate AUC on permuted test set
            new_err = roc_auc_score(data.y_test, y_prob)
                
            # Calculate feature importance (% change in err)
            imp_vals.append((-(new_err - base_err) / base_err) * 100)
        importances.append(imp_vals)
        
    return importances


def main(args):
    mpl.use("Agg")
        
    # Set device for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Set data file path
    test_dir = os.path.join(args.data_dir, "Test")

    # Prepare test data loaders
    test_data = FeatureImpDataset(fpath=os.path.join(args.data_dir, "Test"))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    # Define model
    if args.model == "non-image-only":
        fusion = False
        model = ShallowFFNN(meta_features=test_data.meta_features).to(device)
        # model_name = "110320_shallow-only_ls0.1_seed4"
    elif args.model == "feature-fusion":
        fusion = True
        model = FeatureFusion(meta_features=test_data.meta_features, pre_trained=False, frozen=False).to(device)
        # model_name = "110420_feature-fusion_aug_ls0.1_TTA-5_seed3"
    elif args.model == "learned-feature-fusion":
        fusion = True
        model = LearnedFeatureFusion(meta_features=test_data.meta_features, mode=args.fusion_mode, pre_trained=False, frozen=False).to(device)
        # model_name = "110420_hidden-feature-fusion-concat_aug_ls0.1_TTA-5_seed4"
    elif args.model == "probability-fusion":
        fusion = True
        model = ProbabilityFusion(meta_features=test_data.meta_features, pre_trained=False, frozen=False).to(device)
        # model_name = "110420_probability-fusion_aug_ls0.1_TTA-5_seed4"
    else:
        sys.exit("Invalid model specified.")

    # Load model weights
    model.load_state_dict(torch.load(os.path.join(args.out_dir, args.model_name, args.model_name + ".pt"), map_location=device))
    model.eval()

    # Load shallow feature names
    feature_names_df = pd.read_csv(os.path.join(args.data_dir, "shallow_feature_names.csv"))
    feature_names = feature_names_df["ShallowFeatures"].tolist()

    # Create "mask" to group dummy variables that represent a single feature and rename features accordingly
    feature_mask = list(range(13)) + [[13, 14, 15]] + [[16, 17, 18]] + [19] + [list(range(20, 30))] + [[30, 31, 32]]
    new_feature_names = feature_names[:13]
    new_feature_names.extend(["Parenchymal", "BreastMammoDensity", "FieldStrength", "ScannerModel", "Indication"])

    # Calculate permutation feature importances on test set
    test_importances = permutation_importance(pt_model=model,
                                              data=test_data,
                                              data_loader=test_loader,
                                              feature_mask=feature_mask,
                                              device=device,
                                              fusion=fusion,
                                              iters=args.iters)

    # Rank features by decreasing median importance (across 'iters' runs)
    ranking = np.argsort([np.median(x) for x in test_importances])[::-1]

    # Create box-and-whisker plot of (sorted) feature importance values by feature
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.boxplot([test_importances[i] for i in ranking], vert=False)
    ax.set_yticklabels([new_feature_names[i] for i in ranking])
    ax.set_ylabel("Shallow Feature")
    ax.set_xlabel("Percent Decrease in Test AUC")

    # Create data frame of all feature importance values to save
    imp_df = pd.DataFrame({"Importance": [imp for feature in test_importances for imp in feature],
                        "Feature": [feature for feature in new_feature_names for _ in range(args.iters)],
                        "Iteration": list(range(1, args.iters+1)) * len(new_feature_names)})

    # Save box-and-whisker plot and feature importance values with appropriate names
    fig.savefig(os.path.join(args.out_dir, model_name, f"{get_date()}_{model_name}_feature-imp.pdf"), bbox_inches="tight")
    imp_df.to_csv(os.path.join(args.out_dir, model_name, f"{get_date()}_feature_imp.csv"), index=False)

if __name__ == "__main__":
    # Parse indication argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/mnt/research/midi_lab/holste_data/BreastMRIData_Oct2020/ProcessedData_102820", type=str,
                        help="path to processed data directory (output of preprocess.py)")
    parser.add_argument("--out_dir", default="/mnt/home/holstegr/MIDI/BreastMRIFusionCNN/Results", type=str,
                        help="path to directory where results and model weights are saved")
    parser.add_argument("--model", default="non-image-only", type=str, help="must be one of ['non-image-only', 'feature-fusion', 'learned-feature-fusion', 'probability-fusion']")
    parser.add_argument("--model_name", type=str, help="name of model (e.g., name of saved weights file <model_name>.pt)")
    parser.add_argument("--fusion_mode", default="concat", help="fusion operation for LearnedFeatureFusion or Probability Fusion (one of ['concat', 'multiply', 'add'])")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size for testing")
    parser.add_argument("--iters", default=30, type=int, help="number of times to permute each feature")
    
    args = parser.parse_args()

    # Ensure "--model" argument is valid
    assert (args.model in ['non-image-only', 'feature-fusion', 'learned-feature-fusion', 'probability-fusion']), "--model must be one of ['non-image-only', 'feature-fusion', 'learned-feature-fusion', 'probability-fusion']"

    main(args)