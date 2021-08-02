get_best_run <- function(model_name) {
  # Function that gets directory name of "best" run of
  # 'model_name' by greatest validation AUC
  val_auc <- c()
  for (i in 0:4) {
    if (i == 0)
      path <- paste0(data_dir, model_name, "/", "train_history.csv")
    else 
      path <- paste0(data_dir, model_name, "_seed", i, "/", "train_history.csv")
    val_auc <- c(val_auc, max(read.csv(path)["val_auc_roc"]))
  }
  
  best_seed <- which.max(val_auc) - 1
  if (best_seed != 0)
    return(paste0(model_name, "_seed", best_seed))
  return(model_name)
}

get_self_ensemble_preds <- function(model_name) {
  path <- paste0(data_dir, model_name, "/", "preds.csv")
  pred_df <- read.csv(path)[, c(2, 1)]
  
  for (i in 1:4) {
    path <- paste0(data_dir, model_name, "_seed", i, "/", "preds.csv")
    pred_df[, i+2] <- read.csv(path)["y_prob"]
  }
  
  return(data.frame(y_true=pred_df$y_true,
                    y_prob=rowMeans(pred_df[, 2:ncol(pred_df)])))
}

roc_analysis <- function(model_name, ensemble=F) {
  if (ensemble == T)
    preds <- get_self_ensemble_preds(model_name)
  else {
    # Get run with highest validation AUC to proceed with
    best_model_name <- get_best_run(model_name)
    
    # Read in test set predictions from "best" run
    preds <- read.csv(paste0(data_dir, best_model_name, "/", "preds.csv"))
  }
    
  # Get ROC curve and CIs for AUC and specificity @ 95% sensitivity
  roc_res <- roc(preds$y_true, preds$y_prob)
  set.seed(0)
  auc_ci <- ci(roc_res, of="auc", method="bootstrap", boot.n=5000, boot.stratified=T)
  set.seed(0)
  sp_ci <- ci(roc_res, of="sp", sensitivities=0.95, method="bootstrap",
              boot.n=5000, boot.stratified=T, conf.level=0.95)
  
  # Clean up metrics for printing
  auc <- round(roc_res$auc[1], 3)
  auc_lb <- round(auc_ci[1], 3)
  auc_ub <- round(auc_ci[3], 3)
  sp <- round(max(with(coords(roc_res, transpose=F), specificity[sensitivity >= 0.95])), 3)
  sp_lb <- round(sp_ci[1], 3)
  sp_ub <- round(sp_ci[3], 3)
  
  if (ensemble == T)
    print(paste0("-----", model_name, " (5-run ensemble)-----"))
  else
    print(paste0("-----", model_name, "-----"))
  print(paste0("AUC: ", auc, " [", auc_lb, ", ", auc_ub, "]"))
  print(paste0("Sp @ 95% sens: ", sp, " [", sp_lb, ", ", sp_ub, "]"))
  
  return(roc_res)
}

roc_analysis_ensemble <- function(roc_curves) {
  # TAKE LIST OF ROC_CURVES AND AVERAGE PREDICTIONS
  pred_df <- data.frame(y_true=roc_curves[[1]]$response,
                        y_prob=roc_curves[[1]]$predictor)
  for (i in 2:length(roc_curves)) {
    pred_df[, i+1] <- roc_curves[[i]]$predictor
  }
  
  preds <- data.frame(y_true=pred_df$y_true,
                      y_prob=rowMeans(pred_df[, 2:ncol(pred_df)]))

  
  # Get ROC curve and CIs for AUC and specificity @ 95% sensitivity
  roc_res <- roc(preds$y_true, preds$y_prob)
  set.seed(0)
  auc_ci <- ci(roc_res, of="auc", method="bootstrap", boot.n=5000, boot.stratified=T)
  set.seed(0)
  sp_ci <- ci(roc_res, of="sp", sensitivities=0.95, method="bootstrap",
              boot.n=5000, boot.stratified=T, conf.level=0.95)
  
  # Clean up metrics for printing
  auc <- round(roc_res$auc[1], 3)
  auc_lb <- round(auc_ci[1], 3)
  auc_ub <- round(auc_ci[3], 3)
  sp <- round(max(with(coords(roc_res, transpose=F), specificity[sensitivity >= 0.95])), 3)
  sp_lb <- round(sp_ci[1], 3)
  sp_ub <- round(sp_ci[3], 3)
  
  print(paste0("----- Ensemble -----"))
  print(paste0("AUC: ", auc, " [", auc_lb, ", ", auc_ub, "]"))
  print(paste0("Sp @ 95% sens: ", sp, " [", sp_lb, ", ", sp_ub, "]"))
  
  return(roc_res)
}

get_p_vals <- function(roc_list, model_name_list, metric="auc") {
  if (!(metric %in% c("auc", "sp")))
    return("metric must be 'auc' or 'sp'")
  
  p_vals <- c()
  model_names1 <- c()
  model_names2 <- c()
  model_combs <- combn(length(roc_list), 2)
  for (c in 1:ncol(model_combs)) {
    i <- model_combs[1, c]
    j <- model_combs[2, c]
    
    set.seed(0)
    if (metric == "auc") {
      sig_test <- roc.test(roc_list[[i]], roc_list[[j]], method="bootstrap",
                           boot.n=5000, boot.stratified=T, paired=T)
    } else if (metric == "sp") {
      sig_test <- roc.test(roc_list[[i]], roc_list[[j]], method="sensitivity", sensitivity=c(0.95),
                           boot.n=5000, boot.stratified=T, paired=T)
    } else
      return("Invalid metric specified")
    
    print(paste0("-----", model_name_list[i], " v. ", model_name_list[j], "-----"))
    print(paste0("P = ", sig_test$p.value))
    # print(paste0("Adj. P = ", sig_test$p.value * ncol(model_combs), " (Bonferonni)"))
    
    p_vals <- c(p_vals, sig_test$p.value)
    model_names1 <- c(model_names1, model_name_list[i])
    model_names2 <- c(model_names1, model_name_list[j])
  }
  
  return(list(p_vals=p_vals, model1=model_names1, model2=model_names2))
}

library(pROC)

# Establish paths to results directories
data_dir <- "results/"
model_names <- c("110320_image-only_aug_ls0.1_TTA-5",
                 "110320_shallow-only_ls0.1",
                 "110420_feature-fusion_aug_ls0.1_TTA-5",
                 "110420_hidden-feature-fusion-concat_aug_ls0.1_TTA-5",
                 "110420_probability-fusion_aug_ls0.1_TTA-5")

# Generate ROC curves for best run of each model
roc_curves <- lapply(model_names, roc_analysis)

# # Generate ROC curves for five-run ensemble of each model
# ens_roc_curves <- lapply(model_names, roc_analysis, ensemble=T)

# Generate P values for test of difference in model AUCs
p_values <- get_p_vals(roc_curves, model_names)

# # Generate P values for test of different in five-run ensemble AUCs
# ens_p_values <- get_p_vals(ens_roc_curves, model_names)

# Generate P values for test of difference in model specificities at 95% sensitivity
sp_p_values <- get_p_vals(roc_curves, model_names, metric="sp")

# # Generate P values for test of difference in five-run ensemble specificities at 95% sensitivity
# sp_ens_p_values <- get_p_vals(ens_roc_curves, model_names, metric="sp")