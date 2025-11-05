import pandas as pd
import numpy as np
import zipfile
import os
import json


def unpack_json_zip(zip_path, extract_folder, log_file="unpack_json_errors.csv"):
    """
    Unpack a zip file containing JSON files into a folder and load all JSONs into a DataFrame.
    Adds PortfolioID and ApplicationID columns extracted from the filename.
    Skips invalid or empty JSONs but logs warnings.
    Returns the DataFrame.
    """
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            if not filename:
                continue  # skip directories
            source = zip_ref.open(member)
            target_path = os.path.join(extract_folder, filename)
            with open(target_path, "wb") as target:
                with source:
                    target.write(source.read())
    from tqdm import tqdm
    data_list = []
    portfolio_ids = []
    application_ids = []
    log_entries = []
    json_files = [f for f in os.listdir(extract_folder) if f.endswith('.json')]
    for filename in tqdm(json_files, desc='Extracting JSON files'):
        # Extract PortfolioID and ApplicationID from filename
        # Example: DC_85863453_input_attribute.json
        parts = filename.split('_')
        if len(parts) >= 2:
            portfolio_id = parts[0]
            app_id = parts[1]
        else:
            print(f"[WARNING] Skipping file with invalid name: {filename}")
            log_entries.append({
                "Application_ID": None,
                "PortfolioID": None,
                "filename": filename,
                "error_type": "invalid_filename"
            })
            continue  # skip files not matching pattern
        try:
            with open(os.path.join(extract_folder, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_list.append(data)
                portfolio_ids.append(portfolio_id)
                application_ids.append(app_id)
        except Exception as e:
            error_type = "empty" if str(e) == "Empty file" else "fail"
            print(f"[WARNING] {filename} skipped ({error_type}): {e}")
            log_entries.append({
                "Application_ID": app_id,
                "PortfolioID": portfolio_id,
                "filename": filename,
                "error_type": error_type
            })

    df = pd.DataFrame(data_list)
    # Add PortfolioID and Application_ID columns
    df['PortfolioID'] = portfolio_ids
    df['Application_ID'] = application_ids
    for col in ["i01_BankAccountNum", "i01_BankABA", "i01_Zip"]:
        if col in df.columns:
            df[col] = df[col].astype(str) # to prevent issues with leading zeros
    # Save log if there were any issues
    if log_entries:
        pd.DataFrame(log_entries).to_csv(log_file, index=False)

    return df
    

def save_csv_safe(df, path):
    # Save with utf-8 encoding, fallback to utf-8-sig if error
    try:
        df.to_csv(path, index=False, encoding='utf-8')
    except Exception:
        df.to_csv(path, index=False, encoding='utf-8-sig')


def add_target_label(df, target_path, target_col):
    target = pd.read_csv(target_path, dtype={"Application_ID": str, "PortfolioID": int})
    df["Application_ID"] = df["Application_ID"].astype(str)
    df["PortfolioID"] = df["PortfolioID"].map({'MM': 1, 'RC': 5, 'UW': 6, 'DC': 7})

    if target_col == 'isGood':
        target['isGood'] = target['Payin_120days'].apply(lambda x: 1 if x >= 1.2 else 0)

    df = df.merge(target[["Application_ID", "PortfolioID", target_col]], on=["Application_ID", "PortfolioID"], how="left")
    null_rows = df[target_col].isnull().sum()
    print(f"Null target rows: {null_rows} ({null_rows/len(df)*100:.2f}%)")
    df = df[~df[target_col].isnull()].copy()
    ### HARD CODED DROP FOR ROW WITH INVALID INPUT
    # Hard code invalid rows by Application_ID, PortfolioID
    to_drop = [
        {"Application_ID": "130965435", "PortfolioID": 5},
        {"Application_ID": "104342064", "PortfolioID": 7},
    ]
    for cond in to_drop:
        mask = (df["Application_ID"] == cond["Application_ID"]) & (df["PortfolioID"] == cond["PortfolioID"])
        if mask.any():
            print(f"Dropped row with Application_ID {cond['Application_ID']} and PortfolioID {cond['PortfolioID']} due to invalid input.")
            df = df.loc[~mask]    # drop rows
        else:
            print(f"No rows found for {cond}")
    return df


def stratified_split(df, date_col, test_size=0.2, cutoff_date=None):
    """
    Split dataset into train/test by either:
      - test_size: last fraction of records (default)
      - cutoff_date: explicit datetime cutoff
    
    Args:
        df (pd.DataFrame): input dataframe
        date_col (str): column name of datetime field
        test_size (float): fraction of dataset for test (ignored if cutoff_date is provided)
        cutoff_date (str | datetime, optional): explicit date cutoff for test
    
    Returns:
        train (pd.DataFrame), test (pd.DataFrame)
    """
    # ensure date col is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    if cutoff_date is not None:
        cutoff_date = pd.to_datetime(cutoff_date)
        train = df[df[date_col] < cutoff_date].copy()
        test = df[df[date_col] >= cutoff_date].copy()
    else:
        n_test = int(np.ceil(len(df) * test_size))
        test = df.iloc[-n_test:].copy()
        train = df.iloc[:-n_test].copy()
    
    return train, test


def process_high_cardinality_features(df):
    '''
    Process high-cardinality features:
    - drop i01_BankAccountNum and i01_BankABA. Drop bank name for now unless more information provided
    - For 'i01_Zip', create a new 'zip' column with the first 3 digits
    '''
    drop_cols = []
    # Drop raw unique IDs and institution name
    for col in ["i01_BankAccountNum", "i01_BankABA", 
                "v02_Institution_Name", "v02_Fedwire_Corresponding_Institution", "i01_City",
                "t03_TCA_ActiveMilitaryDutyAlert", "t03_TCA_Alert",  "t03_TCA_Deceased",  "t03_TCA_NoHit",
                "t04_IC_NumPubRecBkrcy24Mons", "v02_PI5LiteScore"]:
        if col in df.columns:
            drop_cols.append(col)
    # map i01_IsHomeOwner values
    if 'i01_IsHomeOwner' in df.columns:
        homeowner_mapping = {'1': 'Own', '2': 'Rent', '3': 'Other'}
        df['i01_IsHomeOwner'] = df['i01_IsHomeOwner'].astype(str).map(homeowner_mapping).fillna('unknown')
    # use only first 3 digits of zip code
    if 'i01_Zip' in df.columns:
        zip_str = df['i01_Zip'].astype(str).str[:3]
        df['zip'] = zip_str.where(zip_str.str.isdigit(), np.nan)
        drop_cols.append('i01_Zip')
    df.drop(drop_cols, axis=1, inplace=True, errors='ignore')
    # Standardize missing values
    df = df.replace(['', 'NoMatch', 'nan', 'NaN', 'None', 'null', 'N/A', '-'], np.nan)
    return df, drop_cols


def process_dates(df, reference_col="i01_ApplicationDate"):
    """
    Process loan-related date columns:
    - Extract app_year and app_month (categorical).
    - Convert DOB to 'age' at application date.
    - Convert other date columns into absolute durations (days since application date).
    - Drop original raw date columns after processing.

    Handles invalid strings like 'NoMatch' or NaN safely.
    
    Returns:
        df (pd.DataFrame): transformed DataFrame
        dropped_cols (list): list of raw date columns dropped
    """
    # Standardize missing values
    df = df.replace(['', 'NoMatch', 'nan', 'NaN', 'None', 'null', 'N/A', '-'], np.nan)
    date_cols = [
        "i01_NextPayDate", "d02_LastChargeOffDate", "d02_LastInquiryDate", "d02_LastPaymentDate", 
        "d02_ReturnDate", "d02_LastReturnDate", "d02_LastTradelineDate", 
        "d02_SecondLastPaymentDate", "d02_ThirdLastPaymentDate", "v02_Record_Last_Updated"
    ]
    dropped_cols = []
    # create reference date features (application date)
    if reference_col in df.columns:
        ref_date = pd.to_datetime(df[reference_col], errors="coerce")
        # df["app_year"] = ref_date.dt.year.astype("category")
        # df["app_month"] = ref_date.dt.month.astype("category")
        dropped_cols.append(reference_col)
    else:
        return # no ref_date, cannot process date-related data
    # birthdate to age at application date
    if 'i01_DOB' in df.columns:
        # remove age for conv model
        dob = pd.to_datetime(df['i01_DOB'], errors='coerce')
        df['age'] = ((ref_date - dob).dt.days / 365.25).astype(float).round()
        df.drop('i01_DOB', axis=1, inplace=True, errors='ignore')
        dropped_cols.append('i01_DOB')
    # convert other date columns to durations since application date
    for col in date_cols:
        if col in df.columns:
            d = pd.to_datetime(df[col], errors="coerce")
            df[f"{col.lower()}_since_app"] = (d - ref_date).dt.days.abs().astype(float)
            dropped_cols.append(col)
            # df.drop(col, axis=1, inplace=True, errors="ignore")
    
    # drop raw application date (already encoded)
    df.drop(dropped_cols, axis=1, inplace=True, errors="ignore")

    return df, dropped_cols


def convert_object_to_numeric(df, threshold = 0.99, target_col=None, numeric_dict=None):
    skip_cols = ['PortfolioID', 'Application_ID', target_col, 'app_month', 'app_year', 'zip']
    # Standardize missing values
    df = df.replace(['', 'NoMatch', 'nan', 'NaN', 'None', 'null', 'N/A', '-'], np.nan)
    if numeric_dict is not None:
        for num_features in numeric_dict.keys():
            if num_features in df.columns:
                df[num_features] = pd.to_numeric(df[num_features], errors='coerce')
    else:
        numeric_dict = {}
        # Try to convert object columns to numeric where possible
        for col in df.columns:
            if col in skip_cols:
                continue
            total = df[col].notna().sum()
            if total <= 0.3 * len(df): # drop feature if 70% or more values are null
                df.drop(col, axis=1, inplace=True, errors='ignore')
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                # already numeric → keep it
                numeric_dict[col] = "numeric"
                continue
            # Try to convert to numeric
            coerced = pd.to_numeric(df[col], errors='coerce')
            numeric_count = coerced.notna().sum()
            # most of the values can be converted to numeric
            # total not null is not small (so the percentage is meaningful)
            if (numeric_count / total) >= threshold:
                df[col] = coerced
                numeric_dict[col] = "numeric"
    return df, numeric_dict


# def encode_categorical(df, target_col, mappings=None):
#     if mappings is None:
#         mappings = {}
#     skip_cols = ['PortfolioID', 'Application_ID', target_col, 'app_month', 'app_year', 'zip', 'i01_City', 'i01_State']
#     for col in df.select_dtypes(include=['object', 'category']).columns:
#         if col in skip_cols:
#             continue
#         if col not in mappings:
#             df[col], uniques = pd.factorize(df[col])
#             mappings[col] = list(uniques)
#         else:
#             uniques = mappings[col]
#             cat_map = {cat: i for i, cat in enumerate(uniques)}
#             df[col] = df[col].astype(object).map(cat_map).fillna(-1).astype(int)
#     return df, mappings


def encode_categorical(df, target_col=None, categorical_dict=None):
    """
    Prepare dataset for CatBoost:
    - Keep categorical columns as 'category' dtype.
    - Return list of categorical feature names, but skip certain ID-like columns and the target column
    """
    skip_cols = ['PortfolioID', 'Application_ID', target_col]
    dict_exists = categorical_dict is not None
    if categorical_dict is None:
        categorical_dict = {}
        # Identify categorical columns excluding skip_cols
        categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns
                        if col not in skip_cols]
    else:
        categorical_cols = [col for col in categorical_dict.keys()]
    # Explicitly cast to category dtype
    for col in categorical_cols:
        df[col] = df[col].replace([np.nan, 'nan', 'NaN', 'None', 'null', '', 'N/A', '-'], 'unknown').astype(str)
        if not dict_exists:
            cats = df[col].unique().tolist()
            if "unknown" not in cats:
                cats.append("unknown")
            categorical_dict[col] = cats
            df[col] = pd.Categorical(df[col], categories=cats)
        else:
            # ensure categories match existing mapping
            valid_cats = set(categorical_dict[col])
            # df[col] = df[col].where(df[col].isin(valid_cats), 'unknown')
            df[col] = pd.Categorical([x if x in valid_cats else 'unknown' for x in df[col]], 
                                    categories=valid_cats)
    return df, categorical_dict


def feature_selection(df, target_col=None, threshold=0.9, numeric_dict=None):
    '''
    Remove highly correlated features above the given threshold.
    Skip ID-like columns and the target column. Ignored categorical features.
    '''
    skip_cols = ['PortfolioID', 'Application_ID', target_col]
    feature_cols = [col for col in numeric_dict.keys() if col not in skip_cols]
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df.drop(to_drop, axis=1, inplace=True, errors="ignore")
    return df, to_drop


def preprocess_pipeline(input_path, target_path, date_col, output_dir, target_col, test_cutoff_date=None, zip_as_input=True):
    os.makedirs(output_dir, exist_ok=True)
    # If input_path is a zip, use unpack_json_zip, else read as csv
    print("Start loading training data...")
    if zip_as_input:
        if input_path.endswith('.zip'):
            extract_folder = os.path.join(output_dir, 'extracted_jsons')
            print("Extracting zip file...")
            df = unpack_json_zip(input_path, extract_folder)
            save_csv_safe(df, os.path.join(output_dir, 'raw_loaded.csv'))
        else:
            df = pd.read_csv(input_path, dtype=str)
        print("Loaded raw data.")
        # df = pd.read_csv(os.path.join(output_dir, 'raw_loaded.csv'), dtype=str)
        df = add_target_label(df, target_path, target_col=target_col)
        print("Added target labels.")
        save_csv_safe(df, os.path.join(output_dir, 'with_target.csv'))
        # Stratify by split_date before preprocessing
        train, test = stratified_split(df, date_col, cutoff_date=test_cutoff_date)
        print("Train test stratified-split done (based on application date: {}).".format(test_cutoff_date))
        save_csv_safe(train, os.path.join(output_dir, 'train_raw.csv'))
        save_csv_safe(test, os.path.join(output_dir, 'test_raw.csv'))
    else: # assume input_path is a csv file with target column
        train = pd.read_csv(os.path.join(output_dir, 'train_raw.csv'), dtype=str)
        test = pd.read_csv(os.path.join(output_dir, 'test_raw.csv'), dtype=str)
        if target_col == 'isGood':
            add_target_label(df, target_path, target_col)
    print("Training data ready.")
    # Training preprocessing
    print("Start training preprocessing...")
    train, dropped_high_cardinality = process_high_cardinality_features(train)
    train, dropped_dates = process_dates(train)
    train, numeric_dict = convert_object_to_numeric(train, target_col=target_col)
    train, categorical_dict = encode_categorical(train, target_col=target_col)
    train, dropped_correlated = feature_selection(train, target_col=target_col, numeric_dict=numeric_dict)
    save_csv_safe(train, os.path.join(output_dir, 'train_processed.csv'))
    print("Training preprocessing complete.")
    # Save mappings
    dropped = dropped_dates + dropped_high_cardinality + dropped_correlated # all columns dropped from previous steps
    with open(os.path.join(output_dir, 'numeric_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(numeric_dict, f, ensure_ascii=False)
    with open(os.path.join(output_dir, 'categorical_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(categorical_dict, f, ensure_ascii=False)
    with open(os.path.join(output_dir, 'dropped_features.json'), 'w', encoding='utf-8') as f:
        json.dump(dropped, f, ensure_ascii=False)
    all_features = numeric_dict.copy()
    all_features.update(categorical_dict)
    used_features = {k: v for k, v in all_features.items() if k not in dropped_correlated}
    with open(os.path.join(output_dir, 'used_features.json'), 'w', encoding='utf-8') as f:
        json.dump(used_features, f, ensure_ascii=False)
    # # Testing preprocessing
    # print("Start testing preprocessing...")
    # test, _ = process_high_cardinality_features(test)
    # test, _ = process_dates(test)
    # test, _ = convert_object_to_numeric(test, target_col=target_col, numeric_dict=numeric_dict)
    # test, _ = encode_categorical(test, target_col=target_col, categorical_dict=categorical_dict)
    # test = test[[col for col in train.columns if col in test.columns]]
    # save_csv_safe(test, os.path.join(output_dir, 'test_processed.csv'))
    # print('Preprocessing complete.')
    # return train, test, list(categorical_dict.keys())
    # Testing
    print("Start testing preprocessing...")
    test, _ = process_high_cardinality_features(test)
    test, _ = process_dates(test)
    test, test_num_dict = convert_object_to_numeric(test, target_col=target_col, numeric_dict=numeric_dict)
    try:
        assert test_num_dict == numeric_dict
    except AssertionError:
        print("[WARNING] Numeric dict mismatch between train and test.")
    test, _ = encode_categorical(test, target_col=target_col, categorical_dict=categorical_dict)
    test = test[[col for col in train.columns if col in test.columns]]
    save_csv_safe(test, os.path.join(output_dir, 'test_processed.csv'))
    print('Preprocessing complete.')
    return train, test, list(categorical_dict.keys())