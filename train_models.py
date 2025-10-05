# src/train_models.py
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from preprocess import load_data, preprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "model_comparison.csv"
OUTPUT_PNG = ROOT / "model_comparison.png"

def train_random_forest(pre, X_train, y_train):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    pipe = Pipeline([('prep', pre), ('clf', rf)])
    grid = {
        'clf__n_estimators': [150, 300],
        'clf__max_depth': [None, 20],
        'clf__min_samples_leaf': [1, 2]
    }
    gs = GridSearchCV(pipe, grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    t0 = time.perf_counter(); gs.fit(X_train, y_train); t1 = time.perf_counter()
    return gs.best_estimator_, t1 - t0

def train_xgboost(pre, X_train, y_train):
    xgb_model = xgb.XGBClassifier(
        random_state=42, eval_metric='logloss',
        n_jobs=-1, tree_method='hist'
    )
    pipe = Pipeline([('prep', pre), ('clf', xgb_model)])
    grid = {
        'clf__n_estimators': [200, 400],
        'clf__max_depth': [4, 8],
        'clf__learning_rate': [0.05, 0.1],
        'clf__subsample': [0.8, 1.0]
    }
    gs = GridSearchCV(pipe, grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    t0 = time.perf_counter(); gs.fit(X_train, y_train); t1 = time.perf_counter()
    return gs.best_estimator_, t1 - t0

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {'name': name, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'cm': cm, 'fpr': fpr, 'tpr': tpr}

def main():
    print("== IN6227 Data Mining – Assignment 1 ==")
    train, test = load_data()
    pre, X_train, X_test, y_train, y_test = preprocess(train, test)

    rf, rf_t = train_random_forest(pre, X_train, y_train)
    xgb, xgb_t = train_xgboost(pre, X_train, y_train)

    rf_res = evaluate(rf, X_test, y_test, 'RandomForest')
    xgb_res = evaluate(xgb, X_test, y_test, 'XGBoost')

    df = pd.DataFrame([
        {'Model':'RandomForest', 'Accuracy':rf_res['acc'], 'Precision':rf_res['prec'], 'Recall':rf_res['rec'], 'F1':rf_res['f1'], 'AUC':rf_res['auc'], 'Train(s)':rf_t},
        {'Model':'XGBoost', 'Accuracy':xgb_res['acc'], 'Precision':xgb_res['prec'], 'Recall':xgb_res['rec'], 'F1':xgb_res['f1'], 'AUC':xgb_res['auc'], 'Train(s)':xgb_t},
    ])
    df.to_csv(OUTPUT_CSV, index=False)

    plt.figure(figsize=(6,6))
    plt.plot(rf_res['fpr'], rf_res['tpr'], label=f"RF (AUC={rf_res['auc']:.3f})")
    plt.plot(xgb_res['fpr'], xgb_res['tpr'], label=f"XGB (AUC={xgb_res['auc']:.3f})")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    print("Saved:", OUTPUT_CSV.name, OUTPUT_PNG.name)

if __name__ == "__main__":
    main()
