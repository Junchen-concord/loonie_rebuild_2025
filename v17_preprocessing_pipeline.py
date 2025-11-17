import os, json
import numpy as np
import pandas as pd

def stratified_split(df, date_col, test_size=0.2, cutoff_date=None):
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    if cutoff_date is not None:
        cutoff_date = pd.to_datetime(cutoff_date)
        train = df[df[date_col] < cutoff_date].copy()
        test  = df[df[date_col] >= cutoff_date].copy()
    else:
        n_test = int(np.ceil(len(df) * test_size))
        test = df.iloc[-n_test:].copy()
        train = df.iloc[:-n_test].copy()
    return train, test

def convert_object_to_numeric(df, target_col=None, threshold=0.99, numeric_dict=None):
    skip_cols = ['PortfolioID', 'Application_ID', target_col, 'app_month', 'app_year', 'zip', 'ApplicationDate']
    df = df.replace(['', 'NoMatch', 'nan', 'NaN', 'None', 'null', 'N/A', '-'], np.nan)
    if numeric_dict is not None:
        for col in numeric_dict.keys():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        numeric_dict = {}
        for col in df.columns:
            if col in skip_cols:
                continue
            total = df[col].notna().sum()
            if total <= 0.3 * len(df):  # drop if ≥70% nulls
                df.drop(col, axis=1, inplace=True, errors='ignore')
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_dict[col] = "numeric"
                continue
            coerced = pd.to_numeric(df[col], errors='coerce')
            numeric_count = coerced.notna().sum()
            if total > 0 and (numeric_count / total) >= threshold:
                df[col] = coerced
                numeric_dict[col] = "numeric"
    return df, numeric_dict

def encode_categorical(df, target_col=None, categorical_dict=None):
    # IBV engineered features are typically numeric; keep API consistent (return empty mapping if none)
    skip_cols = ['PortfolioID', 'Application_ID', target_col, 'ApplicationDate']
    if categorical_dict is None:
        categorical_dict = {}
        categorical_cols = [c for c in df.select_dtypes(include=['object','category']).columns if c not in skip_cols]
    else:
        categorical_cols = list(categorical_dict.keys())
    for col in categorical_cols:
        df[col] = df[col].astype(str).replace(
            [np.nan, 'nan', 'NaN', 'None', 'null', '', 'N/A', '-'], 'unknown'
        )
        if col not in categorical_dict:
            cats = pd.Series(df[col]).astype('category').cat.categories.tolist()
            if "unknown" not in cats:
                cats.append("unknown")
            categorical_dict[col] = cats
        valid = set(categorical_dict[col])
        df[col] = pd.Categorical([x if x in valid else 'unknown' for x in df[col]], categories=valid)
    return df, categorical_dict

def feature_selection(df, target_col=None, numeric_dict=None, threshold=0.9):
    skip_cols = ['PortfolioID', 'Application_ID', target_col, 'ApplicationDate']
    feature_cols = [c for c in (numeric_dict or {}).keys() if c not in skip_cols and c in df.columns]
    if not feature_cols:
        return df, []
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    df.drop(to_drop, axis=1, inplace=True, errors="ignore")
    return df, to_drop

def preprocess_pipeline_ibv(df_final, date_col, target_col, output_dir, test_cutoff_date=None, test_size=0.2):
    os.makedirs(output_dir, exist_ok=True)
    # Basic hygiene
    df = df_final.copy()
    df['Application_ID'] = df['Application_ID'].astype(str)
    if 'PortfolioID' not in df.columns:
        df['PortfolioID'] = 7  # set a constant if you don’t have one
    df = df[~df[target_col].isnull()].copy()

    # Split
    train, test = stratified_split(df, date_col=date_col, cutoff_date=test_cutoff_date, test_size=test_size)

    # Train transforms
    train, numeric_dict = convert_object_to_numeric(train, target_col=target_col)
    train, categorical_dict = encode_categorical(train, target_col=target_col)
    train, dropped_correlated = feature_selection(train, target_col=target_col, numeric_dict=numeric_dict)

    # Save artifacts
    (pd.DataFrame(train).to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False))
    with open(os.path.join(output_dir, 'numeric_dict.json'), 'w') as f: json.dump(numeric_dict, f)
    with open(os.path.join(output_dir, 'categorical_dict.json'), 'w') as f: json.dump(categorical_dict, f)
    with open(os.path.join(output_dir, 'dropped_features.json'), 'w') as f: json.dump(dropped_correlated, f)

    # Test transforms (aligned)
    test, _ = convert_object_to_numeric(test, target_col=target_col, numeric_dict=numeric_dict)
    test, _ = encode_categorical(test, target_col=target_col, categorical_dict=categorical_dict)
    test = test[[c for c in train.columns if c in test.columns]]
    (pd.DataFrame(test).to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False))

    # cat_features from categorical_dict (often empty for IBV)
    cat_features = list(categorical_dict.keys())
    return train, test, cat_features