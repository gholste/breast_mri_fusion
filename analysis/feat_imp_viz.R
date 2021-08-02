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

feat_imp_fig <- function(feat_imp_df, ylim=F) {
  # Rename shallow features to match paper
  feat_imp_df$Feature <- feat_imp_df$Feature %>%
    gsub("leftBreastFlag", "Breast Laterality", .) %>%
    gsub("BreastMammoDensity", "Breast Density", .) %>%
    gsub("Parenchymal", "BPE", .) %>%
    gsub("ShiftDays", "Shift Days", .) %>%
    gsub("maxOriginalImage", "MIP Max Intensity", .) %>%
    gsub("origPixelSpacing", "Pixel Dimensions", .) %>%
    gsub("origCropSizeY_mm", "MIP Height", .) %>%
    gsub("origCropSizeX_mm", "MIP Width", .) %>%
    gsub("FlipAngle", "Flip Angle", .) %>%
    gsub("ReconstructionDiameter", "Reconstruction Diameter", .) %>%
    gsub("ImagingFrequency", "Precession Frequency", .) %>%
    gsub("EchoTime", "Echo Time", .) %>%
    gsub("RepetitionTime", "Repitition Time", .) %>%
    gsub("EchoTrainLength", "Echo Train Length", .) %>%
    gsub("FieldStrength", "Field Strength", .) %>%
    gsub("ScannerModel", 'MRI Software Version', .)
  
  # Get median importance for each feature and model
  ranking <- aggregate(Importance ~ Feature + Model, data = feat_imp_df, median)
  
  # Rank features by median ABSOLUTE importance within each model
  ranking <- ranking %>%
    group_by(Model) %>%
    mutate(feature_rank = order(order(abs(Importance), decreasing = T)))
  
  # Calculate rank product of each feature across all models
  overall_ranking <- aggregate(ranking$feature_rank,
                               list(Feature=ranking$Feature),
                               FUN = function(x) prod(x)^(1/length(x)))
  colnames(overall_ranking)[2] <- "RankProduct"
  overall_ranking <- overall_ranking[order(overall_ranking$RankProduct), ]

  # Reorder "Feature" column to be in order of increasing rank order for plotting
  feat_imp_df$Feature <- factor(feat_imp_df$Feature, levels = overall_ranking$Feature)
  
  # Reorder "Model" column to match order elsewhere in paper
  feat_imp_df$Model <- factor(feat_imp_df$Model,
                              levels = c("Learned Feature Fusion", "Feature Fusion", "Probability Fusion", "Non-Image-Only"))

  # Create box and whisker plot
  g <- ggplot(feat_imp_df, aes(x=log(1+Importance), y=Feature, color=Model)) +
    geom_boxplot(outlier.size = 0.5) +
    # scale_x_continuous(breaks=1:25) + 
    theme_linedraw() + 
    theme(legend.position = c(0.85, 0.85), legend.text = element_text(size=9),
          legend.title = element_text(size=11), legend.title.align = 0.5,
          legend.background = element_rect(linetype = "solid", color = "black")) +
    scale_color_brewer(palette = "Set1", guide = guide_legend(reverse = T)) +
    labs(x = "log(1 + Permutation Importance)", y = "Non-Image Feature")
  
  if (ylim == T)
    g <- g + xlim(NA, 10)
  
  g <- g + theme(legend.position = c(0.7, 0.85))
  return(g)
}

library(ggplot2)
library(ggforce)
library(dplyr)
library(tidyr)

# Set results and model names
data_dir <- "results/"
model_names <- c("110320_shallow-only_ls0.1",
                 "110420_probability-fusion_aug_ls0.1_TTA-5",
                 "110420_feature-fusion_aug_ls0.1_TTA-5",
                 "110420_hidden-feature-fusion-concat_aug_ls0.1_TTA-5")

best_model_names <- lapply(model_names, get_best_run)
paths <- paste0(data_dir, best_model_names, "/110620_feature_imp.csv")

# Read in feature importance results
for (i in 1:4) {
  if (i == 1)
    feat_imp <- read.csv(paths[i])
  else
    feat_imp <- rbind(feat_imp, read.csv(paths[i]))
}
feat_imp$Model <- c(rep("Non-Image-Only", 540), rep("Feature Fusion", 540),
                    rep("Learned Feature Fusion", 540), rep("Probability Fusion", 540))

# Create feature important box-and-whisker plot
g <- feat_imp_fig(feat_imp)
g
ggsave(g, height = 6, width = 5, filename = "log_feat_imp.pdf")