import sys
from sklearn.ensemble import RandomForestClassifier
from lib_ski.Test_Classification_Con_Cancer import X_train
sys.path.append('lib/')
from utilsdata import *
sys.path.append('lib_ski/')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc,confusion_matrix
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
def plot_roc_curve(y_true, y_prob, title="ROC Curve", file_name="roc_curve.pdf"):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'{path}/results/{file_name}')
    plt.show()

    return roc_auc
def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'], title="Confusion Matrix", file_name="confusion_matrix.pdf"):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{path}/results/{file_name}')
    plt.show()

    return cm_normalized
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # 获取预测标签
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算并绘制ROC曲线
    print(f"\n{model_name} - ROC Curves:")
    train_auc = plot_roc_curve(y_train, y_train_prob, title=f"{model_name} - Training ROC Curve", file_name="train_roc_curve.pdf")
    test_auc = plot_roc_curve(y_test, y_test_prob, title=f"{model_name} - Testing ROC Curve", file_name="test_roc_curve.pdf")
    # 计算并绘制混淆矩阵
    print(f"\n{model_name} - Confusion Matrices:")
    train_cm = plot_confusion_matrix(y_train, y_train_pred,
                                     title=f"{model_name} - Training Confusion Matrix", file_name="train_confusion_matrix.pdf")
    test_cm = plot_confusion_matrix(y_test, y_test_pred,
                                    title=f"{model_name} - Testing Confusion Matrix", file_name="test_confusion_matrix.pdf")
    # 返回评估指标
    return {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_confusion_matrix': train_cm,
        'test_confusion_matrix': test_cm
    }

if True:
    path = "H:\Proteomic"
    meta = pd.read_csv(f'{path}/data/hGCN/meta_selected.csv', index_col=0)
    matrix = pd.read_csv(f'{path}/data/hGCN/expr_selected.csv', index_col=0).T
    matrix_Degene_sub = DEgene_selected(matrix)
    matrix = matrix_Degene_sub
    meta_sub = meta[meta.CancerType.isin(['HPV_related', 'NHPV'])]
    matrix_sub = matrix.loc[meta_sub.MS_number, :]
    matrix_train, matrix_test, meta_train, meta_test = SplitData(matrix_sub, meta_sub)
    encoder = LabelEncoder()
    label_train = (torch.tensor(encoder.fit_transform(meta_train.CancerType.values), dtype=torch.float))
    encoder = LabelEncoder()
    label_test = (torch.tensor(encoder.fit_transform(meta_test.CancerType.values), dtype=torch.float))
    X_train = matrix_train
    y_train = label_train
    X_test = matrix_test
    y_test = label_test

rf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
# Compute feature importance
feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
top_marker_genes = feature_importance.nlargest(4) # Select top 20 marker genes



# Plot feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x=top_marker_genes.values, y=top_marker_genes.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Marker Genes")
plt.title("Top Marker Genes Identified by Random Forest")
plt.show()

# Evaluate the model
if True:
    path = os.getcwd()
    X_train = matrix_train
    y_train = label_train
    X_test = matrix_test
    y_test = label_test
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

def evaluate_process(gene_combinations):
    if True:
        top_combos_list = gene_combinations
        X_train = matrix_train.loc[:, top_combos_list]
        X_test = matrix_test.loc[:, top_combos_list]
        scaler = StandardScaler()
        X_train_scaled = X_train
        X_test_scaled = X_test
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        clf = LogisticRegressionCV(
            Cs=[25],
            penalty='elasticnet',  # 使用 ElasticNet 正则化
            fit_intercept=False,
            cv=5,
            solver='saga',  # 使用 saga solver，支持 ElasticNet
            n_jobs=1,
            refit=True,
            class_weight='balanced',
            multi_class='ovr',
            max_iter=1000,
            l1_ratios=[0.5]  # 控制 L1 和 L2 的比例，0.5 表示 L1 和 L2 的比例相等
        )

        clf.fit(X_train_scaled, y_train)
        results = evaluate_model(
            model=clf,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            model_name="Elastic Net Logistic Regression"
        )
    if True:
        model = clf
        explainer = shap.KernelExplainer(model.predict, X_test_scaled)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        shap_values = explainer.shap_values(X_test_scaled)
        shap_obj = explainer(X_test_scaled)
        shap.summary_plot(shap_obj, X_test_scaled, plot_type="bar")
        shap.summary_plot(shap_obj, X_test_scaled)
