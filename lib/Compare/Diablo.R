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

