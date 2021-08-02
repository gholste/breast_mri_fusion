import os
from datetime import datetime
from shutil import rmtree
from copy import deepcopy

import numpy as np
import pandas as pd
import argparse
from scipy import io
from tqdm import tqdm
from scipy.stats import mode

from utils import get_date

def standardize(vec, mean, std):
    out = np.array(deepcopy(vec))
    
    return (out - mean) / std

def normalize_image(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def get_one_hot(df):
    out = df.copy()
    model_name_dummies = ['CADstream 3.1.3.306', 'CADstream 4.0.1.401', 'CADstream 4.0.2.549', 'CADstream 4.1.0.204',
                          'CADstream 5.2.6.458', 'CADstream 5.2.7.543', 'CADstream 5.2.7.563', 'SIGNA EXCITE', 'SIGNA HDx', 'Signa HDxt']

    indication_df = pd.get_dummies(df["Indication"], drop_first=True).rename(columns={1.0: "Screening", 2.0: "Diagnostic", 3.0: "KnownCancer"})
    if df["ManufacturerModelName"].unique().size != 11:
        # If a set does not have all scanner models present, simply pad with appropriate amount of zeroes
        scanner_df = pd.DataFrame(np.zeros((out.shape[0], 10)), columns=model_name_dummies)
        scanner_df_tmp = pd.get_dummies(df["ManufacturerModelName"], drop_first=False)
        scanner_df[scanner_df_tmp.columns] = scanner_df_tmp
    else:
        scanner_df = pd.get_dummies(df["ManufacturerModelName"], drop_first=True)
    strength_df = pd.get_dummies(df["MagneticFieldStrength"], drop_first=True).rename(columns={3.0: "3T"})
    density_df = pd.get_dummies(df["BreastMammoDensity"], drop_first=True).rename(columns={2.0: "Fibrogl", 3.0: "Dense", 4.0: "ExtremelyDense"})
    parenchymal_df = pd.get_dummies(df["Parenchymal"], drop_first=True).rename(columns={2.0: "Mild", 3.0: "Moderate", 4.0: "Marked"})

    out = out.join(parenchymal_df).join(density_df).join(strength_df).join(scanner_df).join(indication_df)
    out = out.drop(columns=["Parenchymal", "BreastMammoDensity", "MagneticFieldStrength", "ManufacturerModelName", "Indication"])

    return out 

def get_dataset(set_type):
    assert set_type in ["Train", "Validation", "Test"], "set_type must be one of ['Training', 'Validation', 'Test']"

    # Get BIRADS scores
    birads = np.array([y.item() for y in data[f"{set_type}Meta"]["BIRADS"][0]])

    # Ensure images are (n, h, w, c)
    images = np.transpose(data[f"{set_type}Images"], (2, 0, 1))[..., np.newaxis]

    # Ensure labels are 1=malignant, 0=benign
    labels = np.array([y.item() for y in data[f"{set_type}Meta"]["Known_Class"][0]])
    labels = np.where(labels == 2, 1, 0)[:, np.newaxis]  # convert "high risk" (1) to benign 

    # Get metadata features of interest (some "structured", some "unstructured")
    meta_dict = {"Age": [], "leftBreastFlag": [], "BreastMammoDensity": [], "Parenchymal": [], "Indication": []}
    u_meta_dict = {"ShiftDays": [], "maxOriginalImage": [], "ManufacturerModelName": [], "origPixelSpacing": [],
                   "origCropSizeY_mm": [], "origCropSizeX_mm": [], "FlipAngle": [], "ReconstructionDiameter": [], 
                   "MagneticFieldStrength": [], "ImagingFrequency": [], "EchoTime": [], "RepetitionTime": [], "EchoTrainLength": []}
    metadata_dict = {**meta_dict, **u_meta_dict}
    for key in meta_dict.keys():
        metadata_dict[key] = np.array([x.item() for x in data[f"{set_type}Meta"][key][0]])
    for key in u_meta_dict.keys():
        for x in data[f"{set_type}MetaUnstruct"][key][0]:
            if x.size == 0:
                metadata_dict[key].append(np.nan)
            else:
                metadata_dict[key].append(x.item())
    metadata_df = pd.DataFrame(metadata_dict)

    return images, labels, metadata_df, birads

def impute_dataset(metadata_df, birads_vec, impute_dict, verbose=False):
    out_df = metadata_df.copy()

    # Identify indices of BIRADS 6 cases
    birads6_idx = np.nonzero(birads_vec == 6)[0]

    if verbose:
        print("# 'KnownCancer' before obfuscation:", out_df[out_df["Indication"] == 3.0].shape[0])

    # For BIRADS 6 cases, replace "KnownCancer" indication with "Screening", "Diagnostic", or "Other" at random (uniformly)
    out_df.loc[birads6_idx, "Indication"] = np.random.choice([0.0, 1.0, 2.0], size=birads6_idx.size, replace=True)
    
    if verbose:
        print("# 'KnownCancer' after obfuscation:", out_df[out_df["Indication"] == 3.0].shape[0])
        print("# Missing values before imputation:")
        print(out_df[list(impute_dict.keys())].isnull().sum())

    # Impute missing values according to impute_dict
    for key in impute_dict.keys():
        out_df.loc[out_df[key].isnull(), key] = impute_dict[key]

    if verbose:
        print("# Missing values after imputation:")
        print(out_df[list(impute_dict.keys())].isnull().sum())

    return out_df

def preprocess_metadata(metadata_df, means, stds, feat_names):
    out_df = metadata_df.copy()

    # One-hot-encode categorical variables
    out_df = get_one_hot(out_df)

    for k, ft in enumerate(feat_names):
        out_df[ft] = standardize(out_df[ft], means[k], stds[k])

    return out_df

def summarize_metadata(metadata_df, quant_features, categorical_features):
    for feat in quant_features:
        print(f"{feat}: {round(metadata_df[feat].mean(), 3)} +/- {round(metadata_df[feat].std(), 3)}")
    
    for feat in categorical_features:
        n = lambda x: x.shape[0]
        total = lambda x: metadata_df.shape[0]
        pct = lambda x: f"{round((x.shape[0] / metadata_df.shape[0])*100, 3)}%"

        print(metadata_df.groupby(feat).agg({feat: [n, total, pct]}))


def main(args):
    # Set necessary paths
    save_dir = os.path.join(args.data_dir, f"ProcessedData_{get_date()}")
    train_dir = os.path.join(save_dir, "Train")
    val_dir = os.path.join(save_dir, "Val")
    test_dir = os.path.join(save_dir, "Test")

    if not args.summarize:
        # Create directories
        if os.path.isdir(save_dir):
            print(f"Deleting {save_dir} and contents...")
            rmtree(save_dir)
        
        print("Creating directories for train, val, and test sets...") 
        os.mkdir(save_dir)
        os.mkdir(train_dir)
        os.mkdir(val_dir)
        os.mkdir(test_dir)

    # Load data
    data = io.loadmat(os.path.join(args.data_dir, "PrepData_28Oct2020.mat"))

    # Get train, val, and test sets
    x_train, y_train, meta_train_df, birads_train = get_dataset("Train")
    x_val, y_val, meta_val_df, birads_val         = get_dataset("Validation")
    x_test, y_test, meta_test_df, birads_test     = get_dataset("Test")

    if args.summarize:
        print("# malignant (train):", np.sum(y_train))
        print("# malignant (val):", np.sum(y_val))
        print("# malignant (test):", np.sum(y_test))

    # Get median/mode of features with missing values in the training set (for imputation)
    imp_age         = np.nanmedian(meta_train_df["Age"])
    imp_mammo_dens  = mode(meta_train_df["BreastMammoDensity"])[0].item()
    imp_parenchymal = mode(meta_train_df["Parenchymal"])[0].item()

    imp_dict = {"Age": imp_age, "BreastMammoDensity": imp_mammo_dens, "Parenchymal": imp_parenchymal}
    if args.verbose:
        print(imp_dict)

    # Obfuscate BIRADS 6 cases and impute missing metadata
    imp_meta_train_df = impute_dataset(meta_train_df, birads_train, imp_dict, args.verbose)
    imp_meta_val_df   = impute_dataset(meta_val_df, birads_val, imp_dict, args.verbose)
    imp_meta_test_df  = impute_dataset(meta_test_df, birads_test, imp_dict, args.verbose)

    # Compute mean + standard deviation for all continuous-valued non-image features in training set
    quant_feat_names = ["Age", "ShiftDays", "maxOriginalImage", "origPixelSpacing", "origCropSizeY_mm", "origCropSizeX_mm",
                        "FlipAngle", "ReconstructionDiameter", "ImagingFrequency", "EchoTime", "RepetitionTime", "EchoTrainLength"]
    quant_means = [meta_train_df[ft].mean() for ft in quant_feat_names]
    quant_stds  = [meta_train_df[ft].std() for ft in quant_feat_names]

    # Preprocess non-image features (standardize continuous features + one-hot-encode categorical features)
    proc_imp_meta_train_df = preprocess_metadata(imp_meta_train_df, quant_means, quant_stds, quant_feat_names)
    proc_imp_meta_val_df   = preprocess_metadata(imp_meta_val_df, quant_means, quant_stds, quant_feat_names)
    proc_imp_meta_test_df  = preprocess_metadata(imp_meta_test_df, quant_means, quant_stds, quant_feat_names)

    # Extract values from data frame (convert to numpy)
    proc_imp_meta_train = proc_imp_meta_train_df.values
    proc_imp_meta_val   = proc_imp_meta_val_df.values
    proc_imp_meta_test  = proc_imp_meta_test_df.values

    # Normalize all images to [0, 1] 
    proc_x_train = np.array([normalize_image(x) for x in x_train])
    proc_x_val   = np.array([normalize_image(x) for x in x_val])
    proc_x_test  = np.array([normalize_image(x) for x in x_test])

    if args.summarize:
        print("---Train metadata---")
        summarize_metadata(metadata_df=imp_meta_train_df,
                        quant_features=["Age", "maxOriginalImage", "origPixelSpacing", "origCropSizeY_mm", "origCropSizeX_mm"],
                        categorical_features=["leftBreastFlag", "BreastMammoDensity", "Parenchymal", "Indication", "ManufacturerModelName"])
        print("---Validation metadata---")
        summarize_metadata(metadata_df=imp_meta_val_df,
                        quant_features=["Age", "maxOriginalImage", "origPixelSpacing", "origCropSizeY_mm", "origCropSizeX_mm"],
                        categorical_features=["leftBreastFlag", "BreastMammoDensity", "Parenchymal", "Indication", "ManufacturerModelName"])
        print("---Test metadata---")
        summarize_metadata(metadata_df=imp_meta_test_df,
                        quant_features=["Age", "maxOriginalImage", "origPixelSpacing", "origCropSizeY_mm", "origCropSizeX_mm"],
                        categorical_features=["leftBreastFlag", "BreastMammoDensity", "Parenchymal", "Indication", "ManufacturerModelName"])
    else:
        if args.verbose:
            # Check shapes
            print("Train images shape:", proc_x_train.shape)
            print("Train metadata shape:", proc_imp_meta_train.shape)
            print("Train labels shape:", y_train.shape) 
            print("Validation images shape:", proc_x_val.shape)
            print("Validation metadata shape:", proc_imp_meta_val.shape)
            print("Validation labels shape:", y_val.shape) 
            print("Test images shape:", proc_x_test.shape)
            print("Test metadata shape:", proc_imp_meta_test.shape)
            print("Test labels shape:", y_test.shape) 

            # Check min/max values
            print("Min. train value:", np.min(proc_x_train))
            print("Max. train value:", np.max(proc_x_train))
            print("Min. validation value:", np.min(proc_x_val))
            print("Max. validation value:", np.max(proc_x_val))
            print("Min. test value:", np.min(proc_x_test))
            print("Max. test value:", np.max(proc_x_test))

            # Check missing values in metadata
            print("# NAs in meta_train:", np.sum(np.isnan(proc_imp_meta_train)))
            print("# NAs in meta_val:", np.sum(np.isnan(proc_imp_meta_val)))
            print("# NAs in meta_test:", np.sum(np.isnan(proc_imp_meta_test)))

            # Check sample distributions
            print(f"Training set is {round((np.sum(y_train == 1) / y_train.shape[0]) * 100, 3)}% malignant")
            print(f"Validation set is {round((np.sum(y_val == 1) / y_val.shape[0]) * 100, 3)}% malignant")
            print(f"Test set is {round((np.sum(y_test == 1) / y_test.shape[0]) * 100, 3)}% malignant")

        # Save column names for use later
        print("Saving shallow feature names...")
        pd.DataFrame({"ShallowFeatures": proc_imp_meta_test_df.columns.values}).to_csv(os.path.join(save_dir, "shallow_feature_names.csv"), index=False)

        # Save files
        for i, (x, meta, y) in tqdm(enumerate(zip(proc_x_train, proc_imp_meta_train, y_train)), total=y_train.shape[0], desc="Saving training data"):
            np.save(os.path.join(train_dir, str(i + 1) + "_x.npy"), x)
            np.save(os.path.join(train_dir, str(i + 1) + "_meta.npy"), meta)
            np.save(os.path.join(train_dir, str(i + 1) + "_y.npy"), y)
        for i, (x, meta, y) in tqdm(enumerate(zip(proc_x_val, proc_imp_meta_val, y_val)), total=y_val.shape[0], desc="Saving validation data"):
            np.save(os.path.join(val_dir, str(i + 1) + "_x.npy"), x)
            np.save(os.path.join(val_dir, str(i + 1) + "_meta.npy"), meta)
            np.save(os.path.join(val_dir, str(i + 1) + "_y.npy"), y)
        for i, (x, meta, y) in tqdm(enumerate(zip(proc_x_test, proc_imp_meta_test, y_test)), total=y_test.shape[0], desc="Saving test data"):
            np.save(os.path.join(test_dir, str(i + 1) + "_x.npy"), x)
            np.save(os.path.join(test_dir, str(i + 1) + "_meta.npy"), meta)
            np.save(os.path.join(test_dir, str(i + 1) + "_y.npy"), y)

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/research/midi_lab/holste_data/BreastMRIData_Oct2020", help="path to unprocessed data")
    parser.add_argument("--summarize", action="store_true", default=False, help="whether or not to provide summary of shallow features (WILL NOT SAVE DATA)")
    parser.add_argument("--verbose", action="store_true", default=False, help="whether or not to include print statements during imputation process")
    args = parser.parse_args()

    main(args)