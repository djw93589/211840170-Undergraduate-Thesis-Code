from medmnist import BreastMNIST
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

#二分类

# 1. 数据加载
train_dataset = BreastMNIST(split='train', download=True)
test_dataset = BreastMNIST(split='test', download=True)

X_train = train_dataset.imgs.reshape(len(train_dataset), -1) / 255.0
X_test = test_dataset.imgs.reshape(len(test_dataset), -1) / 255.0
y_train = train_dataset.labels.flatten()
y_test = test_dataset.labels.flatten()

# 2. 模型定义
models = {
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', device='cuda')
}

# 3. 保存路径
save_dir = './model_visualizations/BreastMNIST'
os.makedirs(save_dir, exist_ok=True)

# 4. 训练和评估
results = {}

for name, model in models.items():
    print(f'\nTraining {name}...')
    start = time.time()
    model.fit(X_train, y_train)
    t = time.time() - start

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc_macro = roc_auc_score(y_test, y_score)
    print(f'{name} Accuracy: {acc:.4f}, AUC: {auc_macro:.4f}')
    print(classification_report(y_test, y_pred))

    results[name] = {
        'model': model,
        'accuracy': acc,
        'auc_macro': auc_macro,
        'y_score': y_score,
        'y_pred': y_pred,
        'train_time': t
    }

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f'{name} - Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f'{name}_confusion_matrix.png'))
    plt.close()

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc_macro:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'{name} - ROC Curve')
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(save_dir, f'{name}_roc_curve.png'))
    plt.close()

    # PR 曲线
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    plt.plot(recall, precision, label=f'{name} (AP={ap:.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'{name} - Precision-Recall Curve')
    plt.legend(); plt.grid()
    plt.savefig(os.path.join(save_dir, f'{name}_pr_curve.png'))
    plt.close()

# 5. 总结
print("\n=== Summary ===")
print("{:<15} {:<10} {:<10} {:<10}".format("Model", "AUC", "ACC", "Time(s)"))
for name, r in results.items():
    print("{:<15} {:.4f}     {:.4f}     {:.2f}".format(name, r['auc_macro'], r['accuracy'], r['train_time']))
