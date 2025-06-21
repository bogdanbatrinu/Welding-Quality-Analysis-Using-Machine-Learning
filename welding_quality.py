# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

def load_data(file_path):
    """
    Load welding data from a CSV file.
    Expected columns: 'welding_speed', 'material_property', 'environment_factor', 'defect'
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        return None

def preprocess_data(data):
    """
    Clean and preprocess the data.
    - Handles 'material_property' with one-hot encoding.
    - Normalizes 'welding_speed', 'environment_factor'.
    - Encodes target variable 'defect'.
    - Handles missing values.
    Returns: processed_df, training_means, training_stds, training_columns
    Returns None for all if data is unsuitable.
    """
    if data is None:
        return None, None, None, None

    processed_df = data.copy()

    # Validate 'defect' column values
    expected_defect_values = {'Non-Defective', 'Defective'}
    if 'defect' in processed_df.columns:
        actual_defect_values = set(processed_df['defect'].dropna().unique())
        if not actual_defect_values.issubset(expected_defect_values):
            unexpected_values = actual_defect_values - expected_defect_values
            print(f"Warning: Unexpected values found in 'defect' column: {unexpected_values}. These will be mapped to NaN.")
        processed_df['defect'] = processed_df['defect'].map({'Non-Defective': 0, 'Defective': 1})
    else:
        print("Warning: 'defect' column not found.")
        return None, None, None, None # Cannot proceed without target

    # Handle 'material_property' with one-hot encoding
    categorical_features = ['material_property']
    numeric_features_to_scale = ['welding_speed', 'environment_factor']

    for col in categorical_features:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(str) # Ensure it's string for get_dummies
            try:
                dummies = pd.get_dummies(processed_df[col], prefix=col, dummy_na=False) # Set dummy_na=False
                processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)
            except Exception as e:
                print(f"Error during one-hot encoding for {col}: {e}")
                # If OHE fails, this column might be problematic, consider returning None
        else:
            print(f"Warning: Categorical feature '{col}' not found. Skipping OHE.")

    # Convert purely numeric features and handle potential errors
    for col in numeric_features_to_scale:
        if col in processed_df.columns:
            if not pd.api.types.is_numeric_dtype(processed_df[col]):
                original_non_numeric_count = processed_df[col].apply(lambda x: not isinstance(x, (int, float))).sum()
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                if original_non_numeric_count > 0:
                    print(f"Warning: Column '{col}' was converted to numeric. It originally contained {original_non_numeric_count} non-numeric value(s) which are now NaN(s).")
        else:
            print(f"Warning: Numeric feature '{col}' not found. It will be treated as missing.")
            processed_df[col] = np.nan # Create it as NaN

    # Define all feature columns *after* OHE, excluding the target
    all_feature_columns = [col for col in processed_df.columns if col != 'defect']

    # Drop rows with missing values in features or target
    # Target NaNs could come from unexpected values in original 'defect' column
    # Feature NaNs could come from original NaNs or coercion errors
    processed_df = processed_df.dropna(subset=all_feature_columns + ['defect'])

    if processed_df.empty:
        print("Warning: Data is empty after dropping NaNs (could be due to missing 'defect' or all rows having NaNs in features).")
        return None, None, None, None

    # Calculate means and stds for scaling from the *current* data (which is assumed to be training data here)
    # These will be used to scale this data and saved for scaling new data at prediction time.
    training_means = processed_df[numeric_features_to_scale].mean()
    training_stds = processed_df[numeric_features_to_scale].std()

    # Handle potential zero std if a feature is constant
    if (training_stds == 0).any():
        problematic_cols = training_stds[training_stds == 0].index.tolist()
        print(f"Warning: Standard deviation is zero for columns: {problematic_cols}. These columns will not be scaled effectively or may cause issues (NaNs/infs if divided by zero).")
        # Replace std=0 with 1 to avoid division by zero, effectively not scaling these columns.
        training_stds[training_stds == 0] = 1.0


    # Normalize numerical features
    processed_df[numeric_features_to_scale] = (processed_df[numeric_features_to_scale] - training_means) / training_stds

    # Get final list of feature columns *after* all processing
    # This list is crucial for ensuring consistency at prediction time.
    final_training_columns = [col for col in processed_df.columns if col != 'defect']

    return processed_df, training_means.to_dict(), training_stds.to_dict(), final_training_columns


def preprocess_input_for_prediction(input_df):
    """
    Preprocesses new input data for prediction using saved training parameters.
    input_df: Pandas DataFrame with raw input features.
    """
    try:
        with open('preprocessing_parameters.json', 'r') as f:
            params = json.load(f)
        training_means = params['means']
        training_stds = params['stds']
        training_columns = params['columns'] # These are all feature columns after OHE during training
    except FileNotFoundError:
        print("Error: preprocessing_parameters.json not found. Train the model first to generate this file.")
        return None
    except Exception as e:
        print(f"Error loading preprocessing parameters: {e}")
        return None

    processed_df = input_df.copy()

    # Handle 'material_property' using one-hot encoding
    # Ensure 'material_property' is string type for get_dummies
    if 'material_property' in processed_df.columns:
        processed_df['material_property'] = processed_df['material_property'].astype(str)
        try:
            # Create dummies for the input's material_property column
            # prefix must match the one used in preprocess_data ('material_property')
            current_dummies = pd.get_dummies(processed_df['material_property'], prefix='material_property', dummy_na=False)
            # Drop the original material_property column before adding dummies
            processed_df = processed_df.drop('material_property', axis=1)
            # Add the new dummy columns
            processed_df = pd.concat([processed_df, current_dummies], axis=1)
        except Exception as e:
            print(f"Error during one-hot encoding of input: {e}")
            return None # Or handle more gracefully
    else:
        print("Warning: 'material_property' column missing in input data.")
        # If it's missing, we still need to ensure the final df has columns for it, filled with 0s.

    # Align columns with training_columns:
    # Add missing columns (that were in training, e.g. a specific material_property_X) and fill with 0
    for col in training_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0

    # Remove any columns in processed_df that were not in training_columns
    # (e.g. a new material_property_Y not seen in training)
    # Make sure to only keep columns that were part of the training_columns
    # However, the numeric features to scale might not be in training_columns if they were the base names.
    # training_columns contains the final feature set names, including OHE columns.

    # Identify purely numeric features that need scaling (must exist in training_means/stds)
    numeric_features_to_scale = [col for col in training_means.keys() if col in processed_df.columns]

    for col in numeric_features_to_scale:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        # Normalize using saved training means and stds
        processed_df[col] = (processed_df[col] - training_means[col]) / training_stds[col]

    # Handle any NaNs that might have been introduced (e.g., from to_numeric coercion if input is bad)
    # A simple strategy: fill with 0 (mean after normalization) or use a more sophisticated imputer if needed.
    # For now, let's ensure all training_columns exist. If a numeric feature became all NaN, it might cause issues.
    # If pd.to_numeric results in NaN, and then we normalize, it will remain NaN.
    # The model might handle NaNs, or we might need to impute.
    # For now, let's assume that if a value is NaN after this, it's problematic.
    # Our current training pipeline drops NaNs before training. So, prediction data should also be free of NaNs.

    # Select and order columns to match training data
    # Must handle cases where a numeric feature to scale might be missing from input_df initially
    # but is present in training_columns (e.g. if it was added and filled with 0)
    final_input_features = pd.DataFrame(columns=training_columns)
    for col in training_columns:
        if col in processed_df.columns:
            final_input_features[col] = processed_df[col]
        else:
            # This case should ideally not happen if we add missing columns and fill with 0 above.
            # But as a safeguard, fill with 0 or a known value.
            final_input_features[col] = 0
            print(f"Warning: Column '{col}' expected from training was not in processed input, filled with 0.")

    # Drop rows if any of the final selected columns have NaNs
    # This mimics the dropna behavior in the training preprocessing
    final_input_features = final_input_features.dropna()

    if final_input_features.empty:
        print("Input data is empty after preprocessing for prediction (likely due to NaNs in critical fields).")
        return None

    return final_input_features[training_columns] # Ensure correct feature set and order


def predict_defects(input_data_df):
    """
    Loads the saved model and predicts defects for new input data.
    input_data_df: Pandas DataFrame with columns 'welding_speed', 'material_property', 'environment_factor'.
    Returns a list of predictions ('Defective' or 'Non-Defective').
    """
    try:
        model = joblib.load('welding_defect_model.joblib')
        print("Model loaded successfully for prediction.")
    except FileNotFoundError:
        print("Error: Model file 'welding_defect_model.joblib' not found. Train the model first.")
        return ["Error: Model not found"] * len(input_data_df)
    except Exception as e:
        print(f"Error loading model: {e}")
        return ["Error: Could not load model"] * len(input_data_df)

    # Preprocess the input data
    # Preprocess the input data using the parameters loaded from JSON.
    preprocessed_input = preprocess_input_for_prediction(input_data_df)

    if preprocessed_input is None or preprocessed_input.empty:
        print("Error: Input data is empty after preprocessing. Cannot predict.")
        return ["Error: Invalid input after preprocessing"] * len(input_data_df)

    # Ensure the columns match the order and set the model was trained on.
    # This is implicitly handled if preprocess_input_for_prediction correctly uses training_material_columns
    # or if the model's internal feature names are used (e.g. scikit-learn >= 1.0 stores feature names).
    # For now, we rely on preprocess_input_for_prediction's `expected_features` logic.
    # If model complains about feature names/numbers, this is where to debug.


    try:
        predictions_numeric = model.predict(preprocessed_input)
        predictions_text = ['Defective' if pred == 1 else 'Non-Defective' for pred in predictions_numeric]
        return predictions_text
    except Exception as e:
        # If the model complains about "X has Y features, but Z is expecting A features",
        # it means preprocessed_input columns don't match model's training columns.
        print(f"Error during prediction: {e}")
        print(f"Columns given to model: {preprocessed_input.columns.tolist()}")
        return ["Error: Prediction failed"] * len(input_data_df)


def train_and_evaluate(data): # Original data passed in
    """
    Preprocesses data, trains and evaluates machine learning models.
    Saves the best model and preprocessing parameters.
    Returns a tuple: (best_model, all_metrics_dict).
    """
    processed_df, training_means, training_stds, final_training_columns = preprocess_data(data)

    if processed_df is None or processed_df.empty:
        print("Data is empty or unsuitable after preprocessing in train_and_evaluate. Skipping model training.")
        return None, {}

    # Save preprocessing parameters
    if training_means is not None and training_stds is not None and final_training_columns is not None:
        preprocessing_params = {
            'means': training_means,
            'stds': training_stds,
            'columns': final_training_columns
        }
        try:
            with open('preprocessing_parameters.json', 'w') as f:
                json.dump(preprocessing_params, f, indent=4)
            print("Preprocessing parameters saved to preprocessing_parameters.json")
        except Exception as e:
            print(f"Error saving preprocessing parameters: {e}")
            # Decide if you want to halt training if params can't be saved
            # For now, we'll print error and continue; prediction might fail later if params are needed but not saved.
    else:
        print("Warning: Preprocessing parameters were not generated. Skipping saving them.")


    # Ensure 'defect' column is present for y
    if 'defect' not in processed_df.columns:
        print("Error: 'defect' column missing in processed_df. Cannot proceed with training.")
        return None, {}

    y = processed_df['defect']

    # X should be defined using final_training_columns from processed_df
    # This ensures X uses the one-hot encoded columns and correct numeric features
    X = processed_df[final_training_columns]


    if X.empty or y.empty:
        print("Features (X) or target (y) is empty. Skipping model training.")
        return None, {}

    if len(y.unique()) < 2:
        print("Target variable 'defect' has less than 2 unique classes after preprocessing. Model training may fail or be meaningless.")
        # return None, {} # Potentially return if this is critical

    # Split data
    test_size_val = 0.3 if len(processed_df) * 0.3 >= 2 else (1 / len(processed_df) if len(processed_df) >=2 else 0)
    if test_size_val == 0 and len(processed_df) < 2:
        print("Insufficient data to split. Skipping model training.")
        return None, {}
    elif test_size_val == 0 and len(processed_df) >=2:
        test_size_val = 1 # use one sample for test

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_val, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
    except ValueError as e:
        print(f"Error during train_test_split (e.g. small sample/class imbalance): {e}. Trying without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_val, random_state=42
        )

    if X_train.empty or X_test.empty:
        print("Training or testing set is empty after split. Skipping model training.")
        return None, {}

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42)
    }

    all_metrics = {}
    best_model = None
    best_accuracy = -1

    for model_name, model in models.items():
        print(f"Training {model_name} model...")
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            # Ensure y_test and predictions have the same length and are not empty
            if len(y_test) == 0 or len(predictions) == 0 :
                print(f"Warning: y_test or predictions are empty for {model_name}. Skipping metrics calculation.")
                accuracy = 0
                report = "y_test or predictions were empty."
            elif len(y_test.unique()) < 2 and len(np.unique(predictions)) < 2 and y_test.unique()[0] == np.unique(predictions)[0]:
                # Handle case where all true values and all predictions are the same single class
                # This can happen with very imbalanced or small test sets.
                # accuracy_score and classification_report might behave unexpectedly or warn.
                # We can define accuracy as 1 if they all match, 0 otherwise, or rely on sklearn's handling.
                accuracy = accuracy_score(y_test, predictions)
                report = classification_report(y_test, predictions, zero_division=0)
                print(f"Note: For {model_name}, y_test and predictions consist of a single, matching class value. Metrics might be limited.")
            else:
                accuracy = accuracy_score(y_test, predictions)
                # Pass target_names for better report if y is 0/1
                target_names_report = ['Non-Defective (0)', 'Defective (1)'] if set(y_test.unique()).issubset({0,1}) and set(predictions).issubset({0,1}) else None
                report = classification_report(y_test, predictions, zero_division=0, target_names=target_names_report)

            all_metrics[model_name] = {
                "accuracy": accuracy,
                "classification_report": report
            }

            print(f"{model_name} Performance:")
            print(f"Accuracy: {accuracy:.4f}")
            print(report)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        except Exception as e:
            print(f"Error training or evaluating {model_name}: {e}")
            all_metrics[model_name] = { "accuracy": 0, "classification_report": f"Error: {e}"}

    if best_model:
        best_model_name = type(best_model).__name__
        print(f"Selected best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        try:
            joblib.dump(best_model, 'welding_defect_model.joblib')
            print(f"Best model ({best_model_name}) saved as welding_defect_model.joblib")
        except Exception as e:
            print(f"Error saving model: {e}")
    else:
        print("No model was successfully trained or selected, so no model was saved.")

    return best_model, all_metrics
# Note: The original train_and_evaluate returned (best_model, all_metrics).
# It still does, but now it also saves the preprocessing_params to a file.

def visualize_data(data):
    """
    Visualize welding defect patterns and critical factors influencing quality.
    Saves plots to 'pairplot.png' and 'correlation_heatmap.png'.
    """
    if data.empty:
        print("Data is empty, skipping visualization.")
        return

    # Pairplot of features and target
    if 'defect' in data.columns and not data['defect'].isnull().all():
        try:
            sns.pairplot(data, hue='defect', diag_kind='kde', corner=True) # Added corner=True for better layout
            plt.suptitle("Pairplot of Welding Features", y=1.02)
            plt.savefig("pairplot.png")
            plt.close() # Close the plot to free memory
            print("Pairplot saved as pairplot.png")
        except Exception as e:
            print(f"Error generating or saving pairplot: {e}")
    else:
        print("Skipping pairplot: 'defect' column is missing, empty, or all NaN.")

    # Correlation heatmap
    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=np.number)
    if not numeric_data.empty:
        # Ensure there are at least 2 numeric columns to calculate correlation
        if numeric_data.shape[1] >= 2:
            try:
                corr = numeric_data.corr()
                plt.figure() # Create a new figure to avoid overlap
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f") # Added fmt for annotation formatting
                plt.title("Correlation Heatmap")
                plt.savefig("correlation_heatmap.png")
                plt.close() # Close the plot to free memory
                print("Correlation heatmap saved as correlation_heatmap.png")
            except Exception as e:
                print(f"Error generating or saving correlation heatmap: {e}")
        else:
            print("Skipping correlation heatmap: Less than 2 numeric columns available.")
    else:
        print("Skipping correlation heatmap: No numeric data available.")

# Ensure all necessary imports are at the top (pandas as pd, numpy as np, etc.)

if __name__ == "__main__":
    # File path to the welding dataset
    file_path = 'welding_data.csv'

    # Load data
    raw_welding_data = load_data(file_path) # Renamed to avoid confusion with processed data

    if raw_welding_data is not None:
        # Train and evaluate models. This will also preprocess data internally
        # and save the model and preprocessing_parameters.json
        # The 'welding_data' variable used by visualize_data needs to be the *processed* one.
        # train_and_evaluate calls preprocess_data which returns the processed_df.
        # We need that processed_df for visualization.

        # To get the processed data for visualization, we can call preprocess_data separately first,
        # or modify train_and_evaluate to return it (which it doesn't currently).
        # Let's call preprocess_data separately for clarity for visualization.

        processed_df_for_viz, _, _, _ = preprocess_data(raw_welding_data.copy()) # Get processed data for viz

        if processed_df_for_viz is not None and not processed_df_for_viz.empty:
            visualize_data(processed_df_for_viz) # Visualize based on processed data
        else:
            print("Data for visualization is empty or None after preprocessing. Skipping visualization.")

        # Now, run the full training pipeline which also does preprocessing internally
        print("\n--- Starting Model Training and Evaluation ---")
        best_model, all_metrics = train_and_evaluate(raw_welding_data.copy()) # Pass a fresh copy for training

        if best_model:
            print(f"Best model ({type(best_model).__name__}) selected, saved, and ready for predictions.")
        else:
            print("No best model was selected. Check logs for details.")

        print("\nDetailed metrics for all models:")
        if all_metrics:
            for model_name, metrics in all_metrics.items():
                print(f"--- {model_name} ---")
                # Ensure accuracy is float before formatting, handle if not present
                acc_val = metrics.get('accuracy')
                acc_str = f"{acc_val:.4f}" if isinstance(acc_val, float) else str(acc_val)
                print(f"  Accuracy: {acc_str}")
                print(f"  Classification Report:\n{metrics.get('classification_report', 'N/A')}")
        else:
            print("No metrics were generated.")

        # Test prediction function (optional demonstration)
        print("\n--- Testing Prediction Function ---")
        if best_model and os.path.exists('preprocessing_parameters.json') and os.path.exists('welding_defect_model.joblib'):
            sample_raw_data_dict = {
                'welding_speed': [1.8, 2.1, 0.5],
                'material_property': ['A', 'B', 'NewMaterial'], # 'NewMaterial' tests unseen category
                'environment_factor': [25.0, 30.0, 10.0]
            }
            sample_input_df = pd.DataFrame(sample_raw_data_dict)

            print("Sample input DataFrame for prediction:")
            print(sample_input_df)

            predictions = predict_defects(sample_input_df)
            print("\nPredictions for sample data:")
            for i, prediction in enumerate(predictions):
                print(f"Sample {i+1}: {prediction}")
        else:
            print("Skipping prediction test because model or parameters are not available.")

    else:
        print("Data loading failed. Please check the file path and data integrity.")

    print("\nScript execution finished.")

# Ensure 'os' is imported if using os.path.exists
# Add 'import os' at the top of the script.
