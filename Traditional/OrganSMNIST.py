import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from medmnist import OrganSMNIST	
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import multiprocessing

def train_model_with_timeout(model, X_train, y_train, timeout_sec):
    def worker(return_dict):
        model.fit(X_train, y_train)
        return_dict['model'] = model

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=worker, args=(return_dict,))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f" Training timed out after {timeout_sec} seconds.")
        return None  # 超时返回 None
    return return_dict.get('model', None)

def compute_multiclass_auc(y_true, y_score, classes=None, average='macro'):
    if classes is None:
        classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    auc_score = roc_auc_score(y_true_bin, y_score, average=average, multi_class='ovr')
    return auc_score

# 1. 创建保存图片的文件夹
save_dir = "./model_visualizations/OrganSMNIST"
os.makedirs(save_dir, exist_ok=True)

# 2. 加载数据
train_dataset = OrganSMNIST(split='train', download=True)
test_dataset = OrganSMNIST(split='test', download=True)

X_train = train_dataset.imgs
y_train = train_dataset.labels
X_test = test_dataset.imgs
y_test = test_dataset.labels

# 3. 数据预处理
if len(X_train.shape) == 3:  # 灰度图
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
elif len(X_train.shape) == 4:  # 彩色图
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

X_train_flat = X_train_flat / 255.0
X_test_flat = X_test_flat / 255.0

if y_train.shape[1] == 1:
    y_train = y_train.flatten()
    y_test = y_test.flatten()

classes = np.unique(y_train)
n_classes = len(classes)

# 4. 定义模型
models = {
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='mlogloss',
        device='cuda'
    )
}

# 5. 训练、预测、评估
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    if name == "SVM":
        trained_model = train_model_with_timeout(model, X_train_flat, y_train, timeout_sec=600)
        if trained_model is None:
            print("Skipping SVM due to timeout.")
            continue
        model = trained_model
    else:
        model.fit(X_train_flat, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test_flat)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_flat)
    else:
        y_score = model.decision_function(X_test_flat)

    auc_macro = compute_multiclass_auc(y_test, y_score, classes=classes, average='macro')
    
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'accuracy': acc,
        'train_time': train_time,
        'report': report,
        'y_score': y_score,           
        'auc_macro': auc_macro        
    }

# 6. 可视化并保存到文件夹
for name, res in results.items():
    model = res['model']
    y_pred = res['y_pred']

    print(f"\n=== {name} Visualization ===")

    # (1) 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'{name} - Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f"{name}_confusion_matrix.png"))
    plt.close()

    # (2) Precision-Recall曲线
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = res['y_score']
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"Class {i} (AP={ap:.2f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{name} - Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"{name}_pr_curve.png"))
    plt.close()

    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - ROC Curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"{name}_roc_curve.png"))
    plt.close()
    
print("\n=== Summary ===")
print("{:<15} {:<10} {:<10} {:<15} ".format("Model","AUC (macro)",  "Accuracy", "Train Time (s)" ))
for name, res in results.items():
    print("{:<15} {:.3f} {:.3f} {:<15.2f} ".format(name, res['auc_macro'], res['accuracy'], res['train_time'] ))
