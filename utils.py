import os
import random
import sys
import time
from copy import deepcopy
from operator import lt, gt
from datetime import datetime

import cv2
import numpy as np
import tqdm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from albumentations.augmentations.transforms import Blur, HorizontalFlip, ElasticTransform, RandomScale, Resize, Rotate, RandomContrast
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch.transforms import ToTensor

def get_date():
    now = datetime.now()

    if now.month < 10:
        str_month = "0" + str(now.month)
    else:
        str_month = str(now.month)

    if now.day < 10:
        str_day = "0" + str(now.day)
    else:
        str_day = str(now.day)

    str_year = str(now.year)[2:]

    return str_month + str_day + str_year

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def to_one_hot(vec):
    '''Converts categorical (n,) mask to one-hot-encoded (n, n_classes) mask
    Args:
        mask (ndarray): int or float mask of shape (n,) or (n, 1)
    Returns:
        res (ndarray): binarized mask of shape (n, n_classes)
    '''
    rows = vec.shape[0]
    n_classes = int(vec.max() + 1)

    res = np.zeros((rows, n_classes))

    for x in range(rows):
        idx = int(vec[x])
        res[x, idx] = 1

    return res

def predict_prob(pt_model, data_loader, device, fusion=False):
    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for step, batch in pbar:
            meta = batch["metadata"].to(device)
            if fusion:
                x = batch["image"].to(device)
                y_hat = pt_model.forward(x, meta)
            else:
                y_hat = pt_model.forward(meta)

            if step == 0:
                y_prob = torch.sigmoid(y_hat).cpu().detach().numpy()
            else:
                y_prob = np.concatenate([y_prob, torch.sigmoid(y_hat).cpu().detach().numpy()])
            
    return y_prob

def train(model, train_loader, max_epochs, optim, device, val_loader=None, class_weights=None, early_stopping=None,
          label_smoothing=0, n_TTA=0, fusion=False, meta_only=False):
    assert isinstance(class_weights, torch.Tensor) or class_weights is None, "class_weights must be of type torch.Tensor or None"
    assert isinstance(early_stopping, dict), "early_stopping must be a dict of the form {'patience': 2, 'metric': val_loss, 'mode': min}"

    history = {"epoch": [], "loss": [], "auc_roc": [], "auc_pr": [], "acc": [],
               "val_loss": [], "val_auc_roc": [], "val_auc_pr": [], "val_acc": []}
    
    pos_wt = None if class_weights is None else class_weights[1]/class_weights[0]
    loss_fxn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt, reduction="none")

    if early_stopping is not None:
        if early_stopping["mode"] == "min":
            op = lt
            best_metric = np.inf
        else:
            op = gt
            best_metric = -np.inf
    else:
        early_stopping["patience"] = np.inf  # to ensure part of outer loop satisfied

    best_model_wts = deepcopy(model.state_dict())
    epochs_no_improve = 0
    epoch = 1
    while epoch <= max_epochs and epochs_no_improve <= early_stopping["patience"]:
        model.train()

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), file=sys.stdout, position=0, leave=True)
        pbar.set_description_str(f"Epoch {epoch}")

        n = 0
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        running_loss = 0.
        for step, batch in pbar:
            optim.zero_grad()  # reset gradients to 0

            x = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            y = batch["label"].to(device)

            if fusion:
                y_hat = model.forward(x, meta)  # feed forward
            elif meta_only:
                y_hat = model.forward(meta)
            else:
                y_hat = model.forward(x)
            y_pred = torch.round(torch.sigmoid(y_hat))

            if step == 0:
                y_prob = torch.sigmoid(y_hat).cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

            if label_smoothing != 0:
                loss = loss_fxn(y_hat, y*(1-label_smoothing)+0.5*label_smoothing)
            else:
                loss = loss_fxn(y_hat, y)
            loss.mean().backward()

            optim.step()  # update params
            running_loss += loss.sum().item()
            n += y.shape[0]
            for i in range(y.shape[0]):
                if y[i] == 1 and y_pred[i] == 1:
                    tp += 1
                if y[i] == 1 and y_pred[i] == 0:
                    fn += 1
                if y[i] == 0 and y_pred[i] == 1:
                    fp += 1
                if y[i] == 0 and y_pred[i] == 0:
                    tn += 1

            l = running_loss / n
            a = (tp + tn) / n
            if tp + fp > 0:
                pr = tp / (tp + fp)
            else:
                pr = np.nan
            if tp + fn > 0:
                re = tp / (tp + fn)
            else:
                re = np.nan
            if tn + fp > 0:
                sp = tn / (tn + fp)
            else:
                sp = np.nan


            if tp + fp + fn > 0:
                pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                  "balanced_acc": (re + sp) / 2, "f1": 2 * tp / (2 * tp + fp + fn)})
            else:
                pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                  "balanced_acc": (re + sp) / 2})
            pbar.refresh()
            time.sleep(0.01)

        # Calculate AUC_ROC and AUC_PR for whole training set
        auc_roc = roc_auc_score(y_true, y_prob)
        prs, res, thrs = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(res, prs)
        print("\tAUC ROC:", round(auc_roc,  3), "| AUC PR:", round(auc_pr, 3))

        history["loss"].append(l)
        history["auc_roc"].append(auc_roc)
        history["auc_pr"].append(auc_pr)
        history["acc"].append(a)

        if val_loader is not None:
            model.eval()

            tqdm.tqdm._instances.clear()
            pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), file=sys.stdout, position=0, leave=True)
            pbar.set_description_str(f"VAL Epoch {epoch}")

            n = 0
            tn = 0
            fp = 0
            fn = 0
            tp = 0
            running_loss = 0.
            with torch.no_grad():
                for step, batch in pbar:
                    x = batch["image"].to(device)
                    meta = batch["metadata"].to(device)
                    y = batch["label"].to(device)

                    if fusion:
                        y_hat = model.forward(x, meta)  # feed forward
                    elif meta_only:
                        y_hat = model.forward(meta)
                    else:
                        y_hat = model.forward(x)
                    y_pred = torch.round(torch.sigmoid(y_hat))

                    if step == 0:
                        y_prob = torch.sigmoid(y_hat).cpu().detach()
                        y_true = y.cpu().detach()
                    else:
                        y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                        y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

                    loss = loss_fxn(y_hat, y)

                    running_loss += loss.sum().item()
                    n += y.shape[0]
                    for i in range(y.shape[0]):
                        if y[i] == 1 and y_pred[i] == 1:
                            tp += 1
                        if y[i] == 1 and y_pred[i] == 0:
                            fn += 1
                        if y[i] == 0 and y_pred[i] == 1:
                            fp += 1
                        if y[i] == 0 and y_pred[i] == 0:
                            tn += 1

                    l = running_loss / n
                    a = (tp + tn) / n
                    if tp + fp > 0:
                        pr = tp / (tp + fp)
                    else:
                        pr = np.nan
                    if tp + fn > 0:
                        re = tp / (tp + fn)
                    else:
                        re = np.nan
                    if tn + fp > 0:
                        sp = tn / (tn + fp)
                    else:
                        sp = np.nan

                    if tp + fp + fn > 0:
                        pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                        "balanced_acc": (re + sp) / 2, "f1": 2 * tp / (2 * tp + fp + fn)})
                    else:
                        pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                        "balanced_acc": (re + sp) / 2})
                    pbar.refresh()
                    time.sleep(0.01)

            # Calculate AUC_ROC and AUC_PR for whole training set
            auc_roc = roc_auc_score(y_true, y_prob)
            prs, res, thrs = precision_recall_curve(y_true, y_prob)
            auc_pr = auc(res, prs)
            print("\tVal AUC ROC:", round(auc_roc,  3), "| Val AUC PR:", round(auc_pr, 3))

            history["val_loss"].append(l)
            history["val_auc_roc"].append(auc_roc)
            history["val_auc_pr"].append(auc_pr)
            history["val_acc"].append(a)

        # Increment epoch
        history["epoch"].append(epoch)
        epoch += 1

        # Early stopping with patience
        if early_stopping is not None:            
            if op(history[early_stopping["metric"]][-1], best_metric):
                print(f"\tEARLY STOPPING: {early_stopping['metric']} improved from {best_metric} to {history[early_stopping['metric']][-1]}")
                epochs_no_improve = 0
                best_metric = history[early_stopping["metric"]][-1]
                best_model_wts = deepcopy(model.state_dict())
            else:
                print(f"\tEARLY STOPPING: {early_stopping['metric']} has NOT improved from {best_metric}")
                epochs_no_improve += 1
            
    # Restore weights from best model
    model.load_state_dict(best_model_wts)
        
    return model, pd.DataFrame(history)

def evaluate(model, data_loader, history_df, device, n_TTA=0, fusion=False, meta_only=False):
    mpl.use("Agg")

    ## HISTORY PLOTS ##
    loss_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history_df["epoch"], history_df["loss"])
    ax.plot(history_df["epoch"], history_df["val_loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(['Train', 'Val'], loc='upper right')

    acc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history_df["epoch"], history_df["acc"])
    ax.plot(history_df["epoch"], history_df["val_acc"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(['Train', 'Val'], loc='upper left')

    auc_roc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history_df["epoch"], history_df["auc_roc"])
    ax.plot(history_df["epoch"], history_df["val_auc_roc"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC ROC")
    ax.legend(['Train', 'Val'], loc='upper left')

    ## GET MODEL PREDICTIONS ##
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    pbar.set_description_str(f"Evaluating...")

    with torch.no_grad():
        for step, batch in pbar:
            x = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            y = batch["label"].to(device)

            if fusion:
                if n_TTA > 0:
                    y_pr = torch.sigmoid(torch.stack([model.forward(x[..., i], meta) for i in range(n_TTA)], dim=0)).mean(dim=0)
                else:
                    y_pr = torch.sigmoid(model.forward(x, meta))
            elif meta_only:
                y_pr = torch.sigmoid(model.forward(meta))
            else:
                if n_TTA > 0:
                    y_pr = torch.sigmoid(torch.stack([model.forward(x[..., i]) for i in range(n_TTA)], dim=0)).mean(dim=0)
                else:
                    y_pr = torch.sigmoid(model.forward(x))

            if step == 0:
                y_prob = y_pr.cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, y_pr.cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

    # Convert to numpy
    y_prob, y_true = y_prob.numpy().squeeze(), y_true.numpy().squeeze()

    # Create dataframe of ground truth labels and predicted probabilities
    pred_df = pd.DataFrame({'y_prob': y_prob, 'y_true': y_true})

    # ROC curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)

    roc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC: {round(auc(fpr, tpr), 3)}")
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random (AUC: 0.5)")
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

    # Confusion matrix (at "optimal" threshold)
    optimal_idx = np.argmin(fpr**2 + (1-tpr)**2)
    y_pred = (y_prob >= thr[optimal_idx]).astype(int)
    CM = confusion_matrix(y_true, y_pred)

    cm_plot, ax = plot_confusion_matrix(conf_mat=CM,
                                        show_normed=True,
                                        show_absolute=True,
                                        class_names=["Benign", "Malignant"],
                                        figsize=(6, 6))

    # Bootstrap AUROC
    n_iters = 10000
    boot_aucs = []
    for _ in range(n_iters):
      idx = np.random.choice(range(y_true.size), y_true.size, replace=True)

      if np.unique(y_true[idx]).size < 2:
          continue

      boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    sorted_boot_aucs = np.array(sorted(boot_aucs))
    auc_lb = sorted_boot_aucs[int(.025 * sorted_boot_aucs.size)]
    auc_ub = sorted_boot_aucs[int(.975 * sorted_boot_aucs.size)]

    # Get specificity at 95% sensitivity
    idx_sp = np.argwhere(tpr >= 0.95)[0].item()

    res = ""
    res += f"Results on test set (n={y_true.size}):\n"
    res += "\n"
    res += f"AUROC: {round(roc_auc_score(y_true, y_prob), 3)}\n"
    res += f"95% CI for AUROC: ({round(auc_lb, 3)}, {round(auc_ub, 3)})\n"
    res += f"\tvia bootstrapping (n={n_iters})\n"
    res += f"Specificity at {round(tpr[idx_sp] * 100, 3)}% sensitivity: {1-fpr[idx_sp]}\n"
    res += f"\tThreshold used: {thr[idx_sp]}\n"
    res += f"'Optimal' threshold for confusion matrix: {thr[optimal_idx]}\n"
    res += "-" * 50 + "\n"
    res += "\n"
    res += repr(model) + "\n"
    res += "-" * 50 + "\n"

    return pred_df, loss_plot, acc_plot, auc_roc_plot, cm_plot, roc_plot, res

def train_multiopt(model, train_loader, max_epochs, optim, device, val_loader=None, class_weights=None, early_stopping=None,
          label_smoothing=0, n_TTA=0, fusion=False, meta_only=False):
    assert isinstance(class_weights, torch.Tensor) or class_weights is None, "class_weights must be of type torch.Tensor or None"
    assert isinstance(early_stopping, dict), "early_stopping must be a dict of the form {'patience': 2, 'metric': val_loss, 'mode': min}"

    history = {"epoch": [], "loss": [], "auc_roc": [], "auc_pr": [], "acc": [],
               "val_loss": [], "val_auc_roc": [], "val_auc_pr": [], "val_acc": []}
    
    pos_wt = None if class_weights is None else class_weights[1]/class_weights[0]
    loss_fxn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt, reduction="none")

    opt_i = torch.optim.Adam(list(model.cnn.parameters()) + list(model.cnn.fc.parameters()) + list(model.cnn_cls.parameters()), lr=1e-4)
    opt_m = torch.optim.Adam(list(model.meta_nn.parameters()) + list(model.meta_nn_cls.parameters()), lr=1e-4)
    opt_f = torch.optim.Adam(list(model.classifier.parameters()), lr=1e-4)

    if early_stopping is not None:
        if early_stopping["mode"] == "min":
            op = lt
            best_metric = np.inf
        else:
            op = gt
            best_metric = -np.inf
    else:
        early_stopping["patience"] = np.inf  # to ensure part of outer loop satisfied

    best_model_wts = deepcopy(model.state_dict())
    epochs_no_improve = 0
    epoch = 1
    while epoch <= max_epochs and epochs_no_improve <= early_stopping["patience"]:
        model.train()

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), file=sys.stdout, position=0, leave=True)
        pbar.set_description_str(f"Epoch {epoch}")

        n = 0
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        running_loss = 0.
        for step, batch in pbar:
            x = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            y = batch["label"].to(device)

            if fusion:
                out = model.forward(x, meta)  # feed forward
                y_hat = out['fusion_out']
                y_i, y_m = out['img_out'], out['meta_out']
            elif meta_only:
                y_hat = model.forward(meta)
            else:
                y_hat = model.forward(x)
            y_pred = torch.round(torch.sigmoid(y_hat))

            if step == 0:
                y_prob = torch.sigmoid(y_hat).cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

            # Minimize sum of loss btw image-only preds, non-image-only preds, and fusion preds
            l_f = loss_fxn(y_hat, y*(1-label_smoothing)+0.5*label_smoothing)
            l_i = loss_fxn(y_i, y*(1-label_smoothing)+0.5*label_smoothing)
            l_m = loss_fxn(y_m, y*(1-label_smoothing)+0.5*label_smoothing)
            loss = (l_f + l_i + l_m)

            opt_f.zero_grad()
            l_f.mean().backward(retain_graph=True)
            opt_f.step()

            opt_i.zero_grad()
            l_i.mean().backward()
            opt_i.step()

            opt_m.zero_grad()
            l_m.mean().backward()
            opt_m.step()

            running_loss += loss.sum().item()
            n += y.shape[0]
            for i in range(y.shape[0]):
                if y[i] == 1 and y_pred[i] == 1:
                    tp += 1
                if y[i] == 1 and y_pred[i] == 0:
                    fn += 1
                if y[i] == 0 and y_pred[i] == 1:
                    fp += 1
                if y[i] == 0 and y_pred[i] == 0:
                    tn += 1

            l = running_loss / n
            a = (tp + tn) / n
            if tp + fp > 0:
                pr = tp / (tp + fp)
            else:
                pr = np.nan
            if tp + fn > 0:
                re = tp / (tp + fn)
            else:
                re = np.nan
            if tn + fp > 0:
                sp = tn / (tn + fp)
            else:
                sp = np.nan


            if tp + fp + fn > 0:
                pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                  "balanced_acc": (re + sp) / 2, "f1": 2 * tp / (2 * tp + fp + fn)})
            else:
                pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                  "balanced_acc": (re + sp) / 2})
            pbar.refresh()
            time.sleep(0.01)

        # Calculate AUC_ROC and AUC_PR for whole training set
        auc_roc = roc_auc_score(y_true, y_prob)
        prs, res, thrs = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(res, prs)
        print("\tAUC ROC:", round(auc_roc,  3), "| AUC PR:", round(auc_pr, 3))

        history["loss"].append(l)
        history["auc_roc"].append(auc_roc)
        history["auc_pr"].append(auc_pr)
        history["acc"].append(a)

        if val_loader is not None:
            model.eval()

            tqdm.tqdm._instances.clear()
            pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), file=sys.stdout, position=0, leave=True)
            pbar.set_description_str(f"VAL Epoch {epoch}")

            n = 0
            tn = 0
            fp = 0
            fn = 0
            tp = 0
            running_loss = 0.
            with torch.no_grad():
                for step, batch in pbar:
                    x = batch["image"].to(device)
                    meta = batch["metadata"].to(device)
                    y = batch["label"].to(device)

                    if fusion:
                        out = model.forward(x, meta)  # feed forward
                        y_hat = out['fusion_out']
                        y_i, y_m = out['img_out'], out['meta_out']
                    elif meta_only:
                        y_hat = model.forward(meta)
                    else:
                        y_hat = model.forward(x)
                    y_pred = torch.round(torch.sigmoid(y_hat))

                    if step == 0:
                        y_prob = torch.sigmoid(y_hat).cpu().detach()
                        y_true = y.cpu().detach()
                    else:
                        y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                        y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

                    l_f = loss_fxn(y_hat, y*(1-label_smoothing)+0.5*label_smoothing)
                    l_i = loss_fxn(y_i, y*(1-label_smoothing)+0.5*label_smoothing)
                    l_m = loss_fxn(y_m, y*(1-label_smoothing)+0.5*label_smoothing)

                    loss = (l_f + l_i + l_m)

                    running_loss += loss.sum().item()
                    n += y.shape[0]
                    for i in range(y.shape[0]):
                        if y[i] == 1 and y_pred[i] == 1:
                            tp += 1
                        if y[i] == 1 and y_pred[i] == 0:
                            fn += 1
                        if y[i] == 0 and y_pred[i] == 1:
                            fp += 1
                        if y[i] == 0 and y_pred[i] == 0:
                            tn += 1

                    l = running_loss / n
                    a = (tp + tn) / n
                    if tp + fp > 0:
                        pr = tp / (tp + fp)
                    else:
                        pr = np.nan
                    if tp + fn > 0:
                        re = tp / (tp + fn)
                    else:
                        re = np.nan
                    if tn + fp > 0:
                        sp = tn / (tn + fp)
                    else:
                        sp = np.nan

                    if tp + fp + fn > 0:
                        pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                        "balanced_acc": (re + sp) / 2, "f1": 2 * tp / (2 * tp + fp + fn)})
                    else:
                        pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                        "balanced_acc": (re + sp) / 2})
                    pbar.refresh()
                    time.sleep(0.01)

            # Calculate AUC_ROC and AUC_PR for whole training set
            auc_roc = roc_auc_score(y_true, y_prob)
            prs, res, thrs = precision_recall_curve(y_true, y_prob)
            auc_pr = auc(res, prs)
            print("\tVal AUC ROC:", round(auc_roc,  3), "| Val AUC PR:", round(auc_pr, 3))

            history["val_loss"].append(l)
            history["val_auc_roc"].append(auc_roc)
            history["val_auc_pr"].append(auc_pr)
            history["val_acc"].append(a)

        # Increment epoch
        history["epoch"].append(epoch)
        epoch += 1

        # Early stopping with patience
        if early_stopping is not None:            
            if op(history[early_stopping["metric"]][-1], best_metric):
                print(f"\tEARLY STOPPING: {early_stopping['metric']} improved from {best_metric} to {history[early_stopping['metric']][-1]}")
                epochs_no_improve = 0
                best_metric = history[early_stopping["metric"]][-1]
                best_model_wts = deepcopy(model.state_dict())
            else:
                print(f"\tEARLY STOPPING: {early_stopping['metric']} has NOT improved from {best_metric}")
                epochs_no_improve += 1
            
    # Restore weights from best model
    model.load_state_dict(best_model_wts)
        
    return model, pd.DataFrame(history)

def train_multiloss(model, train_loader, max_epochs, optim, device, val_loader=None, class_weights=None, early_stopping=None,
                    label_smoothing=0, n_TTA=0, fusion=False, meta_only=False):
    assert isinstance(class_weights, torch.Tensor) or class_weights is None, "class_weights must be of type torch.Tensor or None"
    assert isinstance(early_stopping, dict), "early_stopping must be a dict of the form {'patience': 2, 'metric': val_loss, 'mode': min}"

    history = {"epoch": [], "loss": [], "auc_roc": [], "auc_pr": [], "acc": [],
               "val_loss": [], "val_auc_roc": [], "val_auc_pr": [], "val_acc": []}
    
    pos_wt = None if class_weights is None else class_weights[1]/class_weights[0]
    loss_fxn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_wt, reduction="none")

    if early_stopping is not None:
        if early_stopping["mode"] == "min":
            op = lt
            best_metric = np.inf
        else:
            op = gt
            best_metric = -np.inf
    else:
        early_stopping["patience"] = np.inf  # to ensure part of outer loop satisfied

    best_model_wts = deepcopy(model.state_dict())
    epochs_no_improve = 0
    epoch = 1
    while epoch <= max_epochs and epochs_no_improve <= early_stopping["patience"]:
        model.train()

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), file=sys.stdout, position=0, leave=True)
        pbar.set_description_str(f"Epoch {epoch}")

        n = 0
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        running_loss = 0.
        for step, batch in pbar:
            optim.zero_grad()  # reset gradients to 0

            x = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            y = batch["label"].to(device)

            if fusion:
                out = model.forward(x, meta)  # feed forward
                y_hat = out['fusion_out']
                y_i, y_m = out['img_out'], out['meta_out']
            elif meta_only:
                y_hat = model.forward(meta)
            else:
                y_hat = model.forward(x)
            y_pred = torch.round(torch.sigmoid(y_hat))

            if step == 0:
                y_prob = torch.sigmoid(y_hat).cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

            # Minimize sum of loss btw image-only preds, non-image-only preds, and fusion preds
            l_f = loss_fxn(y_hat, y*(1-label_smoothing)+0.5*label_smoothing)
            l_i = loss_fxn(y_i, y*(1-label_smoothing)+0.5*label_smoothing)
            l_m = loss_fxn(y_m, y*(1-label_smoothing)+0.5*label_smoothing)

            loss = (l_f + l_i + l_m)
            loss.mean().backward()

            optim.step()  # update params
            running_loss += loss.sum().item()
            n += y.shape[0]
            for i in range(y.shape[0]):
                if y[i] == 1 and y_pred[i] == 1:
                    tp += 1
                if y[i] == 1 and y_pred[i] == 0:
                    fn += 1
                if y[i] == 0 and y_pred[i] == 1:
                    fp += 1
                if y[i] == 0 and y_pred[i] == 0:
                    tn += 1

            l = running_loss / n
            a = (tp + tn) / n
            if tp + fp > 0:
                pr = tp / (tp + fp)
            else:
                pr = np.nan
            if tp + fn > 0:
                re = tp / (tp + fn)
            else:
                re = np.nan
            if tn + fp > 0:
                sp = tn / (tn + fp)
            else:
                sp = np.nan


            if tp + fp + fn > 0:
                pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                  "balanced_acc": (re + sp) / 2, "f1": 2 * tp / (2 * tp + fp + fn)})
            else:
                pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                  "balanced_acc": (re + sp) / 2})
            pbar.refresh()
            time.sleep(0.01)

        # Calculate AUC_ROC and AUC_PR for whole training set
        auc_roc = roc_auc_score(y_true, y_prob)
        prs, res, thrs = precision_recall_curve(y_true, y_prob)
        auc_pr = auc(res, prs)
        print("\tAUC ROC:", round(auc_roc,  3), "| AUC PR:", round(auc_pr, 3))

        history["loss"].append(l)
        history["auc_roc"].append(auc_roc)
        history["auc_pr"].append(auc_pr)
        history["acc"].append(a)

        if val_loader is not None:
            model.eval()

            tqdm.tqdm._instances.clear()
            pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), file=sys.stdout, position=0, leave=True)
            pbar.set_description_str(f"VAL Epoch {epoch}")

            n = 0
            tn = 0
            fp = 0
            fn = 0
            tp = 0
            running_loss = 0.
            with torch.no_grad():
                for step, batch in pbar:
                    x = batch["image"].to(device)
                    meta = batch["metadata"].to(device)
                    y = batch["label"].to(device)

                    if fusion:
                        out = model.forward(x, meta)  # feed forward
                        y_hat = out['fusion_out']
                        y_i, y_m = out['img_out'], out['meta_out']
                    elif meta_only:
                        y_hat = model.forward(meta)
                    else:
                        y_hat = model.forward(x)
                    y_pred = torch.round(torch.sigmoid(y_hat))

                    if step == 0:
                        y_prob = torch.sigmoid(y_hat).cpu().detach()
                        y_true = y.cpu().detach()
                    else:
                        y_prob = torch.cat([y_prob, torch.sigmoid(y_hat).cpu().detach()], dim=0)
                        y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

                    l_f = loss_fxn(y_hat, y*(1-label_smoothing)+0.5*label_smoothing)
                    l_i = loss_fxn(y_i, y*(1-label_smoothing)+0.5*label_smoothing)
                    l_m = loss_fxn(y_m, y*(1-label_smoothing)+0.5*label_smoothing)

                    loss = (l_f + l_i + l_m)

                    running_loss += loss.sum().item()
                    n += y.shape[0]
                    for i in range(y.shape[0]):
                        if y[i] == 1 and y_pred[i] == 1:
                            tp += 1
                        if y[i] == 1 and y_pred[i] == 0:
                            fn += 1
                        if y[i] == 0 and y_pred[i] == 1:
                            fp += 1
                        if y[i] == 0 and y_pred[i] == 0:
                            tn += 1

                    l = running_loss / n
                    a = (tp + tn) / n
                    if tp + fp > 0:
                        pr = tp / (tp + fp)
                    else:
                        pr = np.nan
                    if tp + fn > 0:
                        re = tp / (tp + fn)
                    else:
                        re = np.nan
                    if tn + fp > 0:
                        sp = tn / (tn + fp)
                    else:
                        sp = np.nan

                    if tp + fp + fn > 0:
                        pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                        "balanced_acc": (re + sp) / 2, "f1": 2 * tp / (2 * tp + fp + fn)})
                    else:
                        pbar.set_postfix({"loss": l, "acc": a, "precision": pr, "recall": re, "specificity": sp,
                                        "balanced_acc": (re + sp) / 2})
                    pbar.refresh()
                    time.sleep(0.01)

            # Calculate AUC_ROC and AUC_PR for whole training set
            auc_roc = roc_auc_score(y_true, y_prob)
            prs, res, thrs = precision_recall_curve(y_true, y_prob)
            auc_pr = auc(res, prs)
            print("\tVal AUC ROC:", round(auc_roc,  3), "| Val AUC PR:", round(auc_pr, 3))

            history["val_loss"].append(l)
            history["val_auc_roc"].append(auc_roc)
            history["val_auc_pr"].append(auc_pr)
            history["val_acc"].append(a)

        # Increment epoch
        history["epoch"].append(epoch)
        epoch += 1

        # Early stopping with patience
        if early_stopping is not None:            
            if op(history[early_stopping["metric"]][-1], best_metric):
                print(f"\tEARLY STOPPING: {early_stopping['metric']} improved from {best_metric} to {history[early_stopping['metric']][-1]}")
                epochs_no_improve = 0
                best_metric = history[early_stopping["metric"]][-1]
                best_model_wts = deepcopy(model.state_dict())
            else:
                print(f"\tEARLY STOPPING: {early_stopping['metric']} has NOT improved from {best_metric}")
                epochs_no_improve += 1
            
    # Restore weights from best model
    model.load_state_dict(best_model_wts)
        
    return model, pd.DataFrame(history)

def evaluate_multiloss(model, data_loader, history_df, device, n_TTA=0, fusion=False, meta_only=False):
    mpl.use("Agg")

    ## HISTORY PLOTS ##
    loss_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history_df["epoch"], history_df["loss"])
    ax.plot(history_df["epoch"], history_df["val_loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(['Train', 'Val'], loc='upper right')

    acc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history_df["epoch"], history_df["acc"])
    ax.plot(history_df["epoch"], history_df["val_acc"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(['Train', 'Val'], loc='upper left')

    auc_roc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history_df["epoch"], history_df["auc_roc"])
    ax.plot(history_df["epoch"], history_df["val_auc_roc"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC ROC")
    ax.legend(['Train', 'Val'], loc='upper left')

    ## GET MODEL PREDICTIONS ##
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    pbar.set_description_str(f"Evaluating...")

    with torch.no_grad():
        for step, batch in pbar:
            x = batch["image"].to(device)
            meta = batch["metadata"].to(device)
            y = batch["label"].to(device)

            if fusion:
                if n_TTA > 0:
                    y_pr = torch.sigmoid(torch.stack([model.forward(x[..., i], meta)['fusion_out'] for i in range(n_TTA)], dim=0)).mean(dim=0)
                else:
                    y_pr = torch.sigmoid(model.forward(x, meta)['fusion_out'])
            elif meta_only:
                y_pr = torch.sigmoid(model.forward(meta))
            else:
                if n_TTA > 0:
                    y_pr = torch.sigmoid(torch.stack([model.forward(x[..., i]) for i in range(n_TTA)], dim=0)).mean(dim=0)
                else:
                    y_pr = torch.sigmoid(model.forward(x))

            if step == 0:
                y_prob = y_pr.cpu().detach()
                y_true = y.cpu().detach()
            else:
                y_prob = torch.cat([y_prob, y_pr.cpu().detach()], dim=0)
                y_true = torch.cat([y_true, y.cpu().detach()], dim=0)

    # Convert to numpy
    y_prob, y_true = y_prob.numpy().squeeze(), y_true.numpy().squeeze()

    # Create dataframe of ground truth labels and predicted probabilities
    pred_df = pd.DataFrame({'y_prob': y_prob, 'y_true': y_true})

    # ROC curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)

    roc_plot, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC: {round(auc(fpr, tpr), 3)}")
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random (AUC: 0.5)")
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

    # Confusion matrix (at "optimal" threshold)
    optimal_idx = np.argmin(fpr**2 + (1-tpr)**2)
    y_pred = (y_prob >= thr[optimal_idx]).astype(int)
    CM = confusion_matrix(y_true, y_pred)

    cm_plot, ax = plot_confusion_matrix(conf_mat=CM,
                                        show_normed=True,
                                        show_absolute=True,
                                        class_names=["Benign", "Malignant"],
                                        figsize=(6, 6))

    # Bootstrap AUROC
    n_iters = 10000
    boot_aucs = []
    for _ in range(n_iters):
      idx = np.random.choice(range(y_true.size), y_true.size, replace=True)

      if np.unique(y_true[idx]).size < 2:
          continue

      boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    sorted_boot_aucs = np.array(sorted(boot_aucs))
    auc_lb = sorted_boot_aucs[int(.025 * sorted_boot_aucs.size)]
    auc_ub = sorted_boot_aucs[int(.975 * sorted_boot_aucs.size)]

    # Get specificity at 95% sensitivity
    idx_sp = np.argwhere(tpr >= 0.95)[0].item()

    res = ""
    res += f"Results on test set (n={y_true.size}):\n"
    res += "\n"
    res += f"AUROC: {round(roc_auc_score(y_true, y_prob), 3)}\n"
    res += f"95% CI for AUROC: ({round(auc_lb, 3)}, {round(auc_ub, 3)})\n"
    res += f"\tvia bootstrapping (n={n_iters})\n"
    res += f"Specificity at {round(tpr[idx_sp] * 100, 3)}% sensitivity: {1-fpr[idx_sp]}\n"
    res += f"\tThreshold used: {thr[idx_sp]}\n"
    res += f"'Optimal' threshold for confusion matrix: {thr[optimal_idx]}\n"
    res += "-" * 50 + "\n"
    res += "\n"
    res += repr(model) + "\n"
    res += "-" * 50 + "\n"

    return pred_df, loss_plot, acc_plot, auc_roc_plot, cm_plot, roc_plot, res