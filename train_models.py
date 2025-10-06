import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
from preprocess import preprocess

# Create output directory
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluate(name, y_true, y_pred, y_score):
    """Calculate all evaluation metrics"""
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "PR_AUC": average_precision_score(y_true, y_score),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def plot_roc_pr_curves(results_data, save_path):
    """Plot ROC and PR curves for all models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    ax1 = axes[0]
    for name, (y_true, y_score) in results_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)
        ax1.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_score:.4f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Precision-Recall Curve
    ax2 = axes[1]
    for name, (y_true, y_score) in results_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap_score = average_precision_score(y_true, y_score)
        ax2.plot(recall, precision, linewidth=2, label=f'{name} (AP={ap_score:.4f})')
    
    baseline = sum(y_true) / len(y_true)
    ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                label=f'Baseline (P={baseline:.4f})')
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Curves saved to: {save_path}")
    plt.close()

def plot_confusion_matrices(results_list, save_path):
    """Plot confusion matrices for all models"""
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, result in enumerate(results_list):
        cm = result['Confusion_Matrix']
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['<=50K', '>50K'],
                    yticklabels=['<=50K', '>50K'])
        ax.set_title(f"{result['Model']} Confusion Matrix", fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to: {save_path}")
    plt.close()

def main():
    print("="*70)
    print("IN6227 Data Mining - Assignment 1")
    print("Census Income Classification")
    print("="*70)
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    preprocessor, X_train_t, X_test_t, y_train, y_test = preprocess()
    
    print(f"Training samples: {X_train_t.shape[0]:,}")
    print(f"Test samples: {X_test_t.shape[0]:,}")
    print(f"Features: {X_train_t.shape[1]:,}")
    print(f"Positive class ratio (train): {y_train.mean():.4f}")
    
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    print(f"Scale pos weight: {scale_pos_weight:.4f}")
    
    results = []
    curves_data = {}
    
    # Train Logistic Regression
    print("\n[2/5] Training Logistic Regression...")
    print("-" * 70)
    
    lr = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    
    t0 = time.time()
    lr.fit(X_train_t, y_train)
    lr_train_time = time.time() - t0
    
    t1 = time.time()
    lr_pred = lr.predict(X_test_t)
    lr_proba = lr.predict_proba(X_test_t)[:, 1]
    lr_pred_time = time.time() - t1
    
    out_lr = evaluate("Logistic Regression", y_test, lr_pred, lr_proba)
    out_lr["Train_Time(s)"] = lr_train_time
    out_lr["Predict_Time(s)"] = lr_pred_time
    results.append(out_lr)
    curves_data["Logistic Regression"] = (y_test.values, lr_proba)
    
    print(f"✓ Training time: {lr_train_time:.4f}s")
    print(f"✓ Prediction time: {lr_pred_time:.4f}s")
    print(f"✓ Test Accuracy: {out_lr['Accuracy']:.4f}")
    print(f"✓ Test ROC-AUC: {out_lr['ROC_AUC']:.4f}")
    
    # Train XGBoost
    print("\n[3/5] Training XGBoost...")
    print("-" * 70)
    
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='binary:logistic',
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=50,  # Moved here for XGBoost 3.x
        n_jobs=-1,
        random_state=42
    )
    
    t0 = time.time()
    xgb.fit(
        X_train_t, y_train,
        eval_set=[(X_test_t, y_test)],
        verbose=False
    )
    xgb_train_time = time.time() - t0
    
    # Check if early stopping was used
    if hasattr(xgb, 'best_iteration'):
        print(f"✓ Best iteration: {xgb.best_iteration}")
    if hasattr(xgb, 'best_score'):
        print(f"✓ Best validation AUC: {xgb.best_score:.4f}")
    else:
        print(f"✓ Training completed with {xgb.n_estimators} iterations")
    
    t1 = time.time()
    xgb_pred = xgb.predict(X_test_t)
    xgb_proba = xgb.predict_proba(X_test_t)[:, 1]
    xgb_pred_time = time.time() - t1
    
    out_xgb = evaluate("XGBoost", y_test, xgb_pred, xgb_proba)
    out_xgb["Train_Time(s)"] = xgb_train_time
    out_xgb["Predict_Time(s)"] = xgb_pred_time
    results.append(out_xgb)
    curves_data["XGBoost"] = (y_test.values, xgb_proba)
    
    print(f"✓ Training time: {xgb_train_time:.4f}s")
    print(f"✓ Prediction time: {xgb_pred_time:.4f}s")
    print(f"✓ Test Accuracy: {out_xgb['Accuracy']:.4f}")
    print(f"✓ Test ROC-AUC: {out_xgb['ROC_AUC']:.4f}")
    
    # Save results
    print("\n[4/5] Saving results...")
    print("-" * 70)
    
    df = pd.DataFrame(results)
    column_order = [
        "Model", "Accuracy", "Precision", "Recall", "F1", 
        "ROC_AUC", "PR_AUC", "Train_Time(s)", "Predict_Time(s)", 
        "Confusion_Matrix"
    ]
    df = df[column_order]
    
    csv_path = OUTPUT_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Metrics saved to: {csv_path}")
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    display_df = df.drop('Confusion_Matrix', axis=1).round(4)
    print(display_df.to_string(index=False))
    print("="*70)
    
    # Generate plots
    print("\n[5/5] Generating plots...")
    print("-" * 70)
    
    curves_path = OUTPUT_DIR / "roc_pr_curves.png"
    plot_roc_pr_curves(curves_data, curves_path)
    
    cm_path = OUTPUT_DIR / "confusion_matrices.png"
    plot_confusion_matrices(results, cm_path)
    
    print("\n" + "="*70)
    print("✓✓✓ ALL TASKS COMPLETED SUCCESSFULLY! ✓✓✓")
    print("="*70)
    print(f"\nOutput files saved in: {OUTPUT_DIR.absolute()}")
    print(f"  - results.csv")
    print(f"  - roc_pr_curves.png")
    print(f"  - confusion_matrices.png")
    print("="*70)

if __name__ == "__main__":
    main()