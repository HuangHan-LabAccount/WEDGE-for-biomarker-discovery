# https://github.com/Ning-310/POC-19
# DPR-CBS-PLR
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import random
import sys
sys.path.append('../lib/')
from utilsdata import *
if True:
    path = "H:\Proteomic"
    meta = pd.read_csv(f'{path}/data/hGCN/meta_selected.csv', index_col=0)
    matrix = pd.read_csv(f'{path}/data/hGCN/expr_selected.csv', index_col=0).T
    df = matrix.columns
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
df = pd.DataFrame(matrix_train)
#CBS
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list
rd=[]
while rd.__len__()<1000:
    rd0=random_int_list(0, 1113, 4)
    if rd0 not in rd:
        rd.append(rd0)
# FBD
best_lis1 = []
best_model = None
best_auc = 0  # Track the best AUC score
for lis in rd:
    X = df.iloc[:, lis].values
    y = label_train
    def Train_fold(X_train, y_train, X_val, y_val, l):
        Clist = [25]
        clf = LogisticRegressionCV(Cs=Clist, penalty=l, fit_intercept=False, cv=5, solver='liblinear', n_jobs=4,
                                   refit=True, class_weight='balanced', multi_class='ovr')
        clf.fit(X_train, y_train)
        result = []
        coef = clf.coef_.ravel()
        result.append(coef.tolist())
        AUC = []
        y_val_scores = clf.predict_proba(X_val)
        y_onehot = pd.get_dummies(y_val)
        for j in range(2):
            AUC.append(roc_auc_score(y_onehot.iloc[:, j], y_val_scores[:, j]))
        joblib.dump(clf, filename=f'{path}/compare_model/POC_19_Classification.model')
        return y_val, y_val_scores, result, AUC


    vals = np.array([])
    y_vals = np.array([])
    y_val_scores = np.array([[]])
    n = 'Model'

    for r in range(20):
        Skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        for i, (train, val) in enumerate(Skf.split(X, y)):
            X_train = X[train]
            y_train = y[train]
            X_val = X[val]
            y_val = y[val]
            y_val, y_val_score, result, AUC = Train_fold(X_train, y_train, X_val, y_val, 'l1')

        lis1 = []
        for w in result:
            for wi, w0 in enumerate(w):
                if w0 != 0:
                    lis1.append(lis[wi])

            X1 = df.iloc[:, lis1].values
            Skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
            avg_auc = []  # To calculate average AUC for current feature combination

            for i, (train, val) in enumerate(Skf.split(X1, y)):
                X_train = X1[train]
                y_train = y[train]
                X_val = X1[val]
                y_val = y[val]
                y_val, y_val_score, result, AUC = Train_fold(X_train, y_train, X_val, y_val, 'l1')
                print(f"Feature combination: {lis1}, AUC: {AUC}")
                avg_auc.append(np.mean(AUC))  # Calculate mean AUC for this fold

            current_avg_auc = np.mean(avg_auc)

            # Update best results if current combination performs better
            if current_avg_auc > best_auc:
                best_auc = current_avg_auc
                best_lis1 = lis1.copy()  # Save best feature list

                # Retrain model with best features and save it
                best_X = df.iloc[:, best_lis1].values
                Skf_best = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
                for i, (train, val) in enumerate(Skf_best.split(best_X, y)):
                    X_train = best_X[train]
                    y_train = y[train]
                    X_val = best_X[val]
                    y_val = y[val]
                    y_val, y_val_score, result, AUC = Train_fold(X_train, y_train, X_val, y_val, 'l1')
                    best_model = joblib.load(f'{path}/compare_model/POC_19_Classification.model')
print(f"Best feature combination: {best_lis1}")
print(f"Best AUC score: {best_auc}")

# best_lis1 [631, 995, 172, 2]
# df.iloc[:,best_lis1].columns
# Index(['FGF2', 'MSH2', 'PRRX2', 'MBD4'], dtype='object')
pd.DataFrame(rd).to_csv("H:/Proteomic/compare_model/POC_19/rd.csv")
pd.DataFrame(df.iloc[:,best_lis1].columns).to_csv("H:/Proteomic/compare_model/POC_19/best_lis1.csv")
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
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
def evaluate_process(gene_combinations):
    if True:
        top_combos_list = gene_combinations
        X_train = matrix_train.loc[:, top_combos_list]
        X_test = matrix_test.loc[:, top_combos_list]
        y_train = label_train
        y_test = label_test
        scaler = StandardScaler()
        X_train_scaled = X_train
        X_test_scaled = X_test
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        clf = LogisticRegressionCV(
            Cs=[25],
            penalty='elasticnet',
            fit_intercept=False,
            cv=5,
            solver='saga',
            n_jobs=1,
            refit=True,
            class_weight='balanced',
            multi_class='ovr',
            max_iter=1000,
            l1_ratios=[0.5]
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