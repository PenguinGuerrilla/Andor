import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# --- 1. Load Data ---
koi_data_path = "exomoons\KOI_data_values_only.csv"
cumulative_data_path = "exomoons\cumulative_2025.05.26_10.09.12.csv"

# Determine lines to skip based on prior inspection
lines_to_skip_koi = 31
lines_to_skip_cumulative = 86

try:
    df_koi_raw = pd.read_csv(koi_data_path, skiprows=lines_to_skip_koi)
    df_cumulative_raw = pd.read_csv(cumulative_data_path, skiprows=lines_to_skip_cumulative)
except FileNotFoundError:
    print("Error: One or both data files not found. Please ensure the files are in the correct path.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Identify Common Features & Select for Model ---
common_columns = list(set(df_koi_raw.columns) & set(df_cumulative_raw.columns))
potential_features = [
    'koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_prad',
    'koi_teq', 'koi_insol', 'koi_model_snr',
    'koi_steff', 'koi_slogg', 'koi_srad',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'ra', 'dec', 'koi_kepmag'
]
features_for_model = [col for col in potential_features if col in common_columns]
print(f"\nSelected {len(features_for_model)} features for modeling from common columns:", features_for_model)

if not features_for_model:
    print("Error: No common features suitable for modeling found. Exiting.")
    exit()

# --- 3. Define Target Variable and Prepare Training Data ---
df_koi = df_koi_raw[features_for_model + ['kepoi_name']].copy()
df_cumulative_for_training_prep = df_cumulative_raw[features_for_model + ['kepoi_name', 'koi_disposition']].copy()
df_koi['target'] = 1
df_positive_samples = df_koi[features_for_model + ['target']].copy()
koi_target_ids = df_koi['kepoi_name'].unique()
df_negative_samples_pool = df_cumulative_for_training_prep[
    (~df_cumulative_for_training_prep['kepoi_name'].isin(koi_target_ids)) &
    (df_cumulative_for_training_prep['koi_disposition'] == 'FALSE POSITIVE')
]
df_negative_samples_pool = df_negative_samples_pool[features_for_model].copy()
df_negative_samples_pool['target'] = 0

if len(df_negative_samples_pool) > len(df_positive_samples) * 2:
    df_negative_samples = df_negative_samples_pool.sample(n=len(df_positive_samples) * 2, random_state=42)
elif len(df_negative_samples_pool) == 0:
    print("Warning: No negative samples found for training.")
    df_negative_samples = pd.DataFrame(columns=df_positive_samples.columns) # Empty dataframe
else:
    df_negative_samples = df_negative_samples_pool

if not df_negative_samples.empty or not df_positive_samples.empty:
    df_train_val = pd.concat([df_positive_samples, df_negative_samples], ignore_index=True)
else:
    print("Error: Cannot create training data as positive or negative samples are empty. Exiting.")
    exit()

# --- 4. Feature Preprocessing (for training/validation set) ---
X = df_train_val[features_for_model]
y = df_train_val['target']
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=features_for_model)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=features_for_model)

# --- 5. Data Splitting (for model training and validation) ---
min_class_count = y.value_counts().min()
test_size = 0.2
stratify_option = y if min_class_count >= 2 else None
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled_df, y, test_size=test_size, random_state=42, stratify=stratify_option
)

# --- 6. Neural Network Model (MLPClassifier) ---
mlp = MLPClassifier(hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=20,
                    learning_rate_init=0.001)

if X_train.empty or y_train.empty:
    print("Error: Training data is empty. Cannot train the model.")
    exit()
mlp.fit(X_train, y_train)

# --- 7. Model Evaluation and Feature Importance ---
if not X_val.empty:
    y_pred_val = mlp.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    print(classification_report(y_val, y_pred_val, zero_division=0))

    print("\nCalculating Permutation Importance on validation set...")
    if len(X_val) < 10:
        n_repeats_perm = max(1, len(X_val) // 2)
        print(f"Warning: Validation set is very small ({len(X_val)} samples). Reducing n_repeats for permutation importance to {n_repeats_perm}.")
    else:
        n_repeats_perm = 10

    if len(X_val) > 1:
        perm_importance = permutation_importance(
            mlp, X_val, y_val, n_repeats=n_repeats_perm, random_state=42, n_jobs=-1
        )
        
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        
        print("\nTop features defining a 'similar planet' (based on Permutation Importance):")
        print("Feature                                 | Importance (Mean Decrease in Accuracy)")
        print("----------------------------------------|---------------------------------------")
        for i in sorted_idx:
            print(f"{X_val.columns[i]:<39} | {perm_importance.importances_mean[i]:.4f} +/- {perm_importance.importances_std[i]:.4f}")

        # === Code to generate the graph ===
        plt.figure(figsize=(10, 8))
        plt.barh(X_val.columns[sorted_idx], perm_importance.importances_mean[sorted_idx],
                 xerr=perm_importance.importances_std[sorted_idx], align='center', alpha=0.8)
        plt.xlabel("Permutation Importance (Mean Decrease in Accuracy)")
        plt.ylabel("Feature")
        plt.title("Feature Importances for Identifying Similar Planets (Validation Set)")
        plt.gca().invert_yaxis() # Display most important at the top
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.savefig("feature_importances.png")
        print("\nFeature importance plot saved as 'feature_importances.png'")
        # === End of graph generation code ===
    else:
        print("Validation set too small to reliably calculate permutation importance.")
else:
    print("\nValidation set is empty. Skipping validation and feature importance calculation.")

print("\n--- Script Finished ---")