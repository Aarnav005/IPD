import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb
from tqdm import tqdm

def load_labels(label_path):
    """Load the dyslexia class labels."""
    return pd.read_csv(label_path)

def process_metrics_files(data_dir, label_df):
    """Process all metrics files and extract the required features."""
    subject_data = {}
    label_df['subject_id'] = label_df['subject_id'].astype(int)
    
    metrics_files = [f for f in os.listdir(data_dir) if f.endswith('_metrics.csv')]
    
    print(f"Found {len(metrics_files)} metrics files")
    
    for file in tqdm(metrics_files, desc="Processing metrics files"):
        try:
            parts = file.split('_')
            subject_id = int(parts[1])
            
            if subject_id in label_df['subject_id'].values:
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                
                # Get the first row which contains trial-level metrics
                trial_row = df.iloc[0]
                
                if subject_id not in subject_data:
                    subject_data[subject_id] = {}
                
                task_key = 'T1' if 'Syllables' in file else 'T4' if 'Meaningful' in file else 'T5'
                
                # Calculate fixation durations statistics from raw data
                if 'sum_fix_dur_trial' in df.columns and 'n_fix_trial' in df.columns:
                    mean_fix_dur = trial_row['sum_fix_dur_trial'] / trial_row['n_fix_trial'] if trial_row['n_fix_trial'] > 0 else 0
                    
                    # Calculate standard deviation of fixations
                    if 'aoi' in df.columns and 'sum_fix_dur_aoi' in df.columns and 'n_fix_aoi' in df.columns:
                        # Calculate individual fixation durations for each AOI
                        aoi_data = df[df['aoi_kind'] == 'subline']  # Only look at subline AOIs for fixation data
                        if not aoi_data.empty and (aoi_data['n_fix_aoi'] > 0).any():
                            # Calculate mean fixation duration per AOI (more stable than individual fixations)
                            aoi_means = aoi_data.apply(
                                lambda x: x['sum_fix_dur_aoi'] / x['n_fix_aoi'] if x['n_fix_aoi'] > 0 else 0, 
                                axis=1
                            )
                            std_fix_dur = aoi_means.std() if len(aoi_means) > 1 else 0
                        else:
                            # Fallback to trial-level metrics
                            std_fix_dur = trial_row.get('std_fix_dur_trial', 0)
                    else:
                        # Fallback to trial-level metrics if AOI data not available
                        std_fix_dur = trial_row.get('std_fix_dur_trial', 0)
                    
                    # Store the calculated metrics
                    subject_data[subject_id].setdefault(f'{task_key}_n_sacc', []).append(trial_row['n_sacc_trial'])
                    subject_data[subject_id].setdefault(f'{task_key}_mean_fix_dur', []).append(mean_fix_dur)
                    subject_data[subject_id].setdefault(f'{task_key}_std_fix_dur', []).append(std_fix_dur)
                    subject_data[subject_id].setdefault(f'{task_key}_max_fix_dur', []).append(trial_row.get('max_fix_dur_trial', 0))
                    subject_data[subject_id].setdefault(f'{task_key}_n_fix', []).append(trial_row['n_fix_trial'])
                else:
                    # Fallback to original behavior if required columns are missing
                    subject_data[subject_id].setdefault(f'{task_key}_n_sacc', []).append(trial_row['n_sacc_trial'])
                    subject_data[subject_id].setdefault(f'{task_key}_mean_fix_dur', []).append(trial_row.get('mean_fix_dur_trial', 0))
                    subject_data[subject_id].setdefault(f'{task_key}_std_fix_dur', []).append(trial_row.get('std_fix_dur_trial', 0))
                    subject_data[subject_id].setdefault(f'{task_key}_max_fix_dur', []).append(trial_row.get('max_fix_dur_trial', 0))
                    subject_data[subject_id].setdefault(f'{task_key}_n_fix', []).append(trial_row.get('n_fix_trial', 0))

        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    features_list = []
    for subject_id, features in subject_data.items():
        processed_features = {'subject_id': subject_id}
        
        for task in ['T1', 'T4', 'T5']:
            for metric in ['n_sacc', 'mean_fix_dur', 'std_fix_dur', 'max_fix_dur', 'n_fix']:
                key = f'{task}_{metric}'
                if key in features and features[key]:
                    processed_features[key] = np.mean(features[key])
                else:
                    processed_features[key] = 0.0

        features_list.append(processed_features)
    
    if not features_list:
        raise ValueError("No valid data was processed. Check if the subject IDs in the data files match those in the labels file.")
    
    features_df = pd.DataFrame(features_list)
    
    text_characteristics = {
        'T1': {'words': 90},
        'T4': {'words': 113},
        'T5': {'words': 139}
    }
    
    for task in ['T1', 'T4', 'T5']:
        if f'{task}_mean_fix_dur' in features_df.columns:
            features_df[f'{task}_mean_fix_dur'] = features_df[f'{task}_mean_fix_dur'] / 1000.0
        if f'{task}_std_fix_dur' in features_df.columns:
            features_df[f'{task}_std_fix_dur'] = features_df[f'{task}_std_fix_dur'] / 1000.0
        if f'{task}_max_fix_dur' in features_df.columns:
            features_df[f'{task}_max_fix_dur'] = features_df[f'{task}_max_fix_dur'] / 1000.0
    
    features_df['n_sacc_trial_mean'] = features_df['T1_n_sacc'] / text_characteristics['T1']['words']
    features_df['duration_ms_std'] = features_df[[f'T1_std_fix_dur', f'T4_std_fix_dur', f'T5_std_fix_dur']].mean(axis=1)
    features_df['n_sacc_trial_sum'] = features_df['T1_n_sacc'] + features_df['T4_n_sacc'] + features_df['T5_n_sacc']
    features_df['duration_ms_max'] = features_df[[f'T1_max_fix_dur', f'T4_max_fix_dur', f'T5_max_fix_dur']].max(axis=1)
    features_df['duration_ms_mean'] = features_df[[f'T1_mean_fix_dur', f'T4_mean_fix_dur', f'T5_mean_fix_dur']].mean(axis=1)
    features_df['n_fix_mean'] = features_df[[f'T1_n_fix', f'T4_n_fix', f'T5_n_fix']].mean(axis=1)
    
    # Return all 6 features for training
    return features_df[['subject_id', 'n_sacc_trial_mean', 'duration_ms_std', 'n_sacc_trial_sum', 'duration_ms_max', 'duration_ms_mean', 'n_fix_mean']]

def prepare_data(features_df, label_df):
    """Merge features with labels and prepare for model training."""
    merged_df = pd.merge(features_df, label_df, on='subject_id')
    X = merged_df.drop(['subject_id', 'class_id', 'label'], axis=1)
    y = merged_df['class_id']
    return X, y, merged_df

def train_and_evaluate(X, y):
    """Train and evaluate an XGBoost classifier using 5-fold stratified cross-validation."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = []
    roc_aucs = []
    
    print("\n--- Running 5-Fold Cross-Validation ---")
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        roc_aucs.append(roc_auc)

        print(f"\n--- Fold {fold+1} Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Dyslexic', 'Dyslexic']))
        
    print("\n--- Aggregated Cross-Validation Metrics ---")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Mean ROC AUC: {np.mean(roc_aucs):.4f}")
    
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6
    )
    final_model.fit(X, y)
    
    return final_model

def predict_from_input(model, feature_columns):
    """Make predictions based on user input."""
    print("\n--- Manual Prediction ---")
    print("Please enter values for the following features:")
    
    input_data = {}
    
    for col in feature_columns:
        while True:
            try:
                value = float(input(f"Enter {col}: "))
                input_data[col] = value
                break
            except ValueError:
                print(f"Invalid input. Please enter a numeric value.")
    
    input_df = pd.DataFrame([input_data])
    
    proba = model.predict_proba(input_df)[0]
    
    threshold = 0.25
    prediction = 1 if proba[1] >= threshold else 0
    
    label = "Dyslexic" if prediction == 1 else "Non-Dyslexic"
    confidence = proba[1] * 100
    
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 25)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "All_data", "data")
    label_path = os.path.join(base_dir, "All_data", "dyslexia_class_label.csv")
    model_path = os.path.join(base_dir, "dyslexia_xgboost_model.json")
    
    if not os.path.exists(data_dir) or not os.path.exists(label_path):
        print("Error: Data directory or label file not found.")
        return
    
    print("Loading labels...")
    label_df = load_labels(label_path)
    
    print("\nProcessing metrics files and extracting features...")
    features_df = process_metrics_files(data_dir, label_df)
    
    # Display first few rows of the features data
    print("\nFirst few rows of the features data:")
    print(features_df.head())
    
    print("\nPreparing data...")
    X, y, merged_df = prepare_data(features_df, label_df)
    
    # Display first few rows of the prepared data
    print("\nFirst few rows of the prepared data (X):")
    print(X.head())
    print("\nFirst few labels (y):")
    print(y.head())
    
    print("\nTraining XGBoost model...")
    final_model = train_and_evaluate(X, y)
    
    final_model.save_model(model_path)
    print(f"\nModel saved to '{model_path}'")

    predict_from_input(final_model, X.columns.tolist())

if __name__ == "__main__":
    main()