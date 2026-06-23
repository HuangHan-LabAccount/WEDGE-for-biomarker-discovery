options(warn=-1)
library(PRROC)
if (!require("mixOmics")) {
  if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
  BiocManager::install("mixOmics")
  library(mixOmics)
}
library(dplyr)
set.seed(42)

# File paths
# proteome_file <- "H:/GOAT2.0-main/raw_data/GEA/expr_selected.csv"
# meta_file <- "H:/GOAT2.0-main/raw_data/GEA/meta_selected.csv"
# 
# 
# head(proteome_raw)
# head(meta_data)

proteome_file <- "H:/Proteomic/data/hGCN/expr_selected.csv"
meta_file <- "H:/Proteomic/data/hGCN/meta_selected_2.csv"

# Read data
proteome_raw <- t(read.csv(proteome_file, row.names=1, header=TRUE, check.names=FALSE))
meta_data <- read.csv(meta_file, header=TRUE)


# Ensure row names in proteome match sample IDs in meta data
rownames(meta_data) <- rownames(proteome_raw)


meta_data <- meta_data[meta_data$CancerType != "Con",]
meta_data$CancerType <- recode(meta_data$CancerType,
                               "NHPV" = 1,
                               "HPV_related" = 0)
proteome_raw <- proteome_raw[rownames(proteome_raw) %in% rownames(meta_data),]
# Split into training and testing sets based on Batch
train_samples <- rownames(meta_data)[meta_data$Batch == 1]
test_samples <- rownames(meta_data)[meta_data$Batch == 2]

# Prepare X and Y for training
X_train <- proteome_raw[train_samples, , drop=FALSE]
Y_train <- as.factor(meta_data[train_samples, "CancerType"])
# Train model using sPLS-DA (simplified from DIABLO as we only have one dataset)
# Since we're only using one omics dataset, we'll use splsda instead of block.splsda
splsda_result <- splsda(X_train, Y_train, keepX = c(50, 2))

# Prepare testing data
X_test <- proteome_raw[test_samples, , drop=FALSE]
Y_test <- as.factor(meta_data[test_samples, "CancerType"])

# Make predictions
predictions <- predict(splsda_result, newdata = X_test)
# Get probabilities for ROC and PR curves
# For class labels 0 and 1, we'll extract probabilities for class 1
probs <- predictions$predict[, 2, 1]  # Component 1, Class 1 probabilities
# Reformat to match your original approach
# Assuming 1 is positive class and 0 is negative class
fg <- probs[Y_test == 1]  # Foreground: probabilities for actual positives
bg <- probs[Y_test == 0]  # Background: probabilities for actual negatives

# $value
# value.var
# MCM3     0.25826407
# MCM4     0.25565838
# MCM6     0.24345463
# MCM7     0.23425363
# MCM2     0.23393019
# GINS1    0.22658342
# MCM5     0.21496985
# HAT1     0.20779776
# TYMS     0.19935965
# RFC4     0.19373010
# DNAJC9   0.19368157
# RFC2     0.18513527
# NASP     0.18279744
# PCNA     0.17198843
# GINS3    0.16802369
# RFC3     0.16431660
# LIG1     0.15774071
# GINS4    0.14709431
# UHRF1    0.14578913
# RFC5     0.14284343
# RBBP7    0.13651756
# MCMBP    0.13566086
# RRM1     0.13314702
# WDR76    0.12057262
# PRIM2    0.10813284
# CDKN2A   0.10720967
# FEN1     0.10318971
# IPO9     0.10264398
# MSH6     0.10000596
# CHTF18   0.08730545
# PBK      0.08670071
# EED      0.08663800
# UBE2T    0.08259892
# DUT      0.08159196
# STMN1    0.08037738
# CDK2     0.07861867
# GINS2    0.06721839
# MSH2     0.06619583
# RBBP4    0.06515932
# DNMT1    0.05835394
# DHFR     0.05720893
# ASPSCR1 -0.05342534
# EXOSC5   0.04413048
# STMN2    0.04163777
# UNG      0.03498981
# SSRP1    0.03330335
# VRK1     0.02397249
# UBR7     0.02374274
# POLD2    0.01763357
# KIF2C    0.01682879