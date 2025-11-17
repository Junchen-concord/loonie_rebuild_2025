from preprocessing_pipeline import preprocess_pipeline
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, roc_auc_score
import pandas as pd 
import numpy as np
import joblib

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

def load_processed_data(train, test, target_col):
    exclude_cols = ['Application_ID', 'PortfolioID', target_col]
    X_train = train.drop(columns=exclude_cols)
    y_train = train[target_col].astype(float).astype(int)
    X_test = test.drop(columns=exclude_cols)
    y_test = test[target_col].astype(float).astype(int)
    return X_train, y_train, X_test, y_test


def bayesian_catboost_train(X_train, y_train, X_test, y_test, model_path, cat_features, param_space, resample_method=None):
    '''
    Bayesian optimization for CatBoostClassifier, saves best model, prints best performance.
    '''
    if resample_method:
        weights = [1.0, 1.0]
    else:
        class_counts = np.bincount(y_train)
        total = len(y_train)
        weights = [total / (2 * c) if c > 0 else 1.0 for c in class_counts]

    print("class weights calculated:\n", weights)

    steps = []
    if resample_method:
        if resample_method == "smote":
            sampler = SMOTE(sampling_strategy=0.3, random_state=42)
        elif resample_method == "under":
            sampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        elif resample_method == "both":
            sampler = ImbPipeline([
                ('over', SMOTE(sampling_strategy=0.3, random_state=42)),
                ('under', RandomUnderSampler(sampling_strategy=0.8, random_state=42))
            ])
        else:
            raise ValueError(f"Unknown resample_method: {resample_method}")
        steps.append(('resample', sampler))

    cb_model = CatBoostClassifier(
        verbose=100,
        class_weights=weights,
        eval_metric='F1',
        random_state=42,
        iterations=700,
        early_stopping_rounds=150
    )

    steps.append(('catboost', cb_model))
    pipeline = ImbPipeline(steps)

    opt = BayesSearchCV(
        estimator=pipeline,
        search_spaces=param_space,
        scoring="f1",
        n_iter=20, 
        cv=3, 
        n_jobs=-1,
        verbose=2
    )

    opt.fit(X_train, y_train, 
        **{"catboost__cat_features": cat_features})

    print("Best parameters found:", opt.best_params_)
    print("Best CV F1:", opt.best_score_)

    best_pipeline = opt.best_estimator_
    best_model = best_pipeline.named_steps['catboost']
    best_model.save_model(model_path, format="json")
    joblib.dump(best_pipeline, model_path.replace(".json", "_pipeline.pkl"))
    print(f"Best model saved to {model_path} and pipeline saved to {model_path.replace('.json', '_pipeline.pkl')}")
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    print('ROC AUC (test data):', roc_auc_score(y_test, y_proba))
    print("Initial classification report (threshold=0.5):")
    test_report = classification_report(y_test, y_pred)
    print(test_report)

    print("Optimized threshold:")
    # Precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    # Compute F1 at all thresholds
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    print(f"Best threshold: {best_threshold:.3f}, "
        f"Precision: {precisions[best_idx]:.3f}, "
        f"Recall: {recalls[best_idx]:.3f}, "
        f"F1: {f1_scores[best_idx]:.3f}")

    # Apply optimized threshold
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    test_report_opt = classification_report(y_test, y_pred_opt)
    print(test_report_opt)

    return best_model, test_report_opt


def train_pipeline(zip_path, target_path, date_col, output_dir, 
                      model_path, target_col, test_cutoff_date, param_space,         
                      resample_method):
    train, test, cat_features = preprocess_pipeline(zip_path, target_path, date_col, output_dir, target_col, test_cutoff_date=test_cutoff_date, zip_as_input=True)
    X_train, y_train, X_test, y_test = load_processed_data(train, test, target_col)
    print("Training features shape:", X_train.shape)
    print("Begin Bayesian CatBoost training...")
    best_model, test_report = bayesian_catboost_train(X_train, y_train, X_test, y_test, model_path, cat_features, param_space, resample_method)
    return best_model, test_report