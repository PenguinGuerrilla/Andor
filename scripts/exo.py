import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

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

print("--- df_koi_raw Initial Load ---")
print(f"Shape: {df_koi_raw.shape}")
# print("Columns:", df_koi_raw.columns.tolist()) # Optional: for brevity
# df_koi_raw.info() # Optional: for brevity

print("\n--- df_cumulative_raw Initial Load ---")
print(f"Shape: {df_cumulative_raw.shape}")
# print("Columns:", df_cumulative_raw.columns.tolist()) # Optional: for brevity
# df_cumulative_raw.info(max_cols=10, verbose=False) # Optional: for brevity

# --- 2. Identify Common Features & Select for Model ---
common_columns = list(set(df_koi_raw.columns) & set(df_cumulative_raw.columns))
# print(f"\nFound {len(common_columns)} common columns initially.")
# print("Common columns:", common_columns) # Optional: for brevity

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
# Create copies to avoid SettingWithCopyWarning
df_koi = df_koi_raw[features_for_model + ['kepoi_name']].copy()
# We need 'koi_disposition' from cumulative for filtering negative samples
df_cumulative_for_training_prep = df_cumulative_raw[features_for_model + ['kepoi_name', 'koi_disposition']].copy()

# Add a 'target' column. Planets from df_koi are our target class (1).
df_koi['target'] = 1
df_positive_samples = df_koi[features_for_model + ['target']].copy()

# For negative examples (0): use 'FALSE POSITIVE' from df_cumulative that are NOT in df_koi
koi_target_ids = df_koi['kepoi_name'].unique()
df_negative_samples_pool = df_cumulative_for_training_prep[
    (~df_cumulative_for_training_prep['kepoi_name'].isin(koi_target_ids)) &
    (df_cumulative_for_training_prep['koi_disposition'] == 'FALSE POSITIVE')
]
df_negative_samples_pool = df_negative_samples_pool[features_for_model].copy()
df_negative_samples_pool['target'] = 0

print(f"\nNumber of positive samples (from df_koi): {len(df_positive_samples)}")
print(f"Number of potential negative samples (FALSE POSITIVE, not in df_koi, from df_cumulative): {len(df_negative_samples_pool)}")

# Balance the dataset: aim for roughly 2x negative samples compared to positive
if len(df_negative_samples_pool) > len(df_positive_samples) * 2:
    df_negative_samples = df_negative_samples_pool.sample(n=len(df_positive_samples) * 2, random_state=42)
elif len(df_negative_samples_pool) == 0:
    print("Warning: No negative samples (FALSE POSITIVE not in KOI list) found. Model training might be problematic.")
    # Handle this case: e.g., use a different strategy for negative samples or stop.
    # For now, we'll proceed, but this is a critical point.
    df_negative_samples = pd.DataFrame(columns=df_positive_samples.columns) # Empty dataframe
else:
    df_negative_samples = df_negative_samples_pool

if not df_negative_samples.empty or not df_positive_samples.empty:
    df_train_val = pd.concat([df_positive_samples, df_negative_samples], ignore_index=True)
else:
    print("Error: Cannot create training data as positive or negative samples are empty. Exiting.")
    exit()

print(f"\nCombined dataset for training/validation shape: {df_train_val.shape}")
print("Target distribution in combined dataset:")
print(df_train_val['target'].value_counts())

# --- 4. Feature Preprocessing (for training/validation set) ---
X = df_train_val[features_for_model]
y = df_train_val['target']

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=features_for_model)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=features_for_model)

# --- 5. Data Splitting (for model training and validation) ---
# Stratify y if there are enough samples in the smallest class
min_class_count = y.value_counts().min()
test_size = 0.2
stratify_option = y if min_class_count >= 2 else None # Stratify if at least 2 samples per class for split

if stratify_option is None and min_class_count < 2 :
    print(f"Warning: Smallest class has only {min_class_count} sample(s). Stratification disabled for train-test split.")

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled_df, y, test_size=test_size, random_state=42, stratify=stratify_option
)

print(f"\nTraining data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

# --- 6. Neural Network Model (MLPClassifier) ---
mlp = MLPClassifier(hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=20,
                    learning_rate_init=0.001) # Explicitly set learning rate

print("\nTraining Neural Network...")
if X_train.empty or y_train.empty:
    print("Error: Training data is empty. Cannot train the model.")
    exit()

mlp.fit(X_train, y_train)

# Evaluate the model
if not X_val.empty:
    y_pred_val = mlp.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    print(classification_report(y_val, y_pred_val, zero_division=0))
else:
    print("\nValidation set is empty. Skipping validation.")


# --- 7. Make Predictions on the Entire Relevant Cumulative Dataset ---
# Prepare the full df_cumulative for prediction:
# We want to predict on 'CONFIRMED' or 'CANDIDATE' planets from df_cumulative_raw.
df_predict_pool = df_cumulative_raw[
    (df_cumulative_raw['koi_disposition'] == 'CONFIRMED') |
    (df_cumulative_raw['koi_disposition'] == 'CANDIDATE')
].copy()

if not df_predict_pool.empty:
    # Ensure all necessary columns for features_for_model are present
    missing_cols_in_predict = set(features_for_model) - set(df_predict_pool.columns)
    if missing_cols_in_predict:
        print(f"Error: The following features are missing in the prediction pool: {missing_cols_in_predict}")
        # Handle this, e.g., by adding NaNs or re-evaluating feature selection
        # For now, we'll exit if critical features are missing for prediction.
        exit()

    X_predict_full = df_predict_pool[features_for_model].copy()

    # Impute and scale using the transformers fitted on the training data
    X_predict_full_imputed = imputer.transform(X_predict_full) # Use transform, not fit_transform
    X_predict_full_imputed_df = pd.DataFrame(X_predict_full_imputed, columns=features_for_model)

    X_predict_full_scaled = scaler.transform(X_predict_full_imputed_df) # Use transform
    X_predict_full_scaled_df = pd.DataFrame(X_predict_full_scaled, columns=features_for_model)

    print(f"\nMaking predictions on {len(X_predict_full_scaled_df)} 'CONFIRMED' or 'CANDIDATE' planets from df_cumulative_raw...")
    predictions_cumulative = mlp.predict(X_predict_full_scaled_df)
    probabilities_cumulative = mlp.predict_proba(X_predict_full_scaled_df)[:, 1] # Probability of class 1 (similar)

    # Add predictions and probabilities to the DataFrame
    # Ensure index alignment if X_predict_full was derived from df_predict_pool
    df_predict_pool['predicted_similar_to_koi'] = predictions_cumulative
    df_predict_pool['similarity_score_to_koi'] = probabilities_cumulative

    # Identify planets predicted as similar (class 1)
    similar_planets_df = df_predict_pool[df_predict_pool['predicted_similar_to_koi'] == 1].copy()
    print(f"\nFound {len(similar_planets_df)} planets in df_cumulative (CONFIRMED/CANDIDATE) predicted as similar to those in df_koi.")

    # Display top 10 most similar planets
    # Ensure all columns for display are present in similar_planets_df
    display_columns = ['kepoi_name', 'koi_disposition', 'similarity_score_to_koi'] + features_for_model
    cols_to_display_final = [col for col in display_columns if col in similar_planets_df.columns]

    print("\nTop 10 most similar planets from df_cumulative (CONFIRMED/CANDIDATE):")
    print(similar_planets_df.sort_values(by='similarity_score_to_koi', ascending=False)[cols_to_display_final].head(10))

    # Save the results to a CSV file
    output_filename = "similar_planets_predictions.csv"
    try:
        similar_planets_df.sort_values(by='similarity_score_to_koi', ascending=False).to_csv(output_filename, index=False)
        print(f"\nFull list of predicted similar planets saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
else:
    print("\nNo 'CONFIRMED' or 'CANDIDATE' planets found in df_cumulative for prediction, or df_predict_pool is empty.")

print("\n--- Script Finished ---")