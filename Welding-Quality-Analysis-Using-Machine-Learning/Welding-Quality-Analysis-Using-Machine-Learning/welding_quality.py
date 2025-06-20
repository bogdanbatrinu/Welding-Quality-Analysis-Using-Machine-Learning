# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    - Handle missing values
    - Encode categorical variables
    - Feature scaling
    """
    # Validate 'defect' column values
    expected_defect_values = {'Non-Defective', 'Defective'}
    actual_defect_values = set(data['defect'].dropna().unique())
    if not actual_defect_values.issubset(expected_defect_values):
        unexpected_values = actual_defect_values - expected_defect_values
        print(f"Warning: Unexpected values found in 'defect' column: {unexpected_values}. These will be mapped to NaN.")

    # Encode the target variable 'defect'
    data['defect'] = data['defect'].map({'Non-Defective': 0, 'Defective': 1})

    # Handle categorical feature 'material_property' using one-hot encoding
    if 'material_property' in data.columns:
        print(f"Original 'material_property' unique values: {data['material_property'].unique()}")
        # Using dummy_na=True to create a separate category for NaNs if they exist after potential pd.to_numeric
        data = pd.get_dummies(data, columns=['material_property'], prefix='mat_prop', dummy_na=True)
        print(f"Data columns after one-hot encoding 'material_property': {data.columns.tolist()}")
    else:
        print("Warning: 'material_property' column not found. Skipping one-hot encoding for it.")

    # Define purely numeric features for conversion and scaling (excluding one-hot encoded ones)
    # 'welding_speed', 'environment_factor' are expected to be numeric.
    numeric_cols_to_normalize = ['welding_speed', 'environment_factor']

    for col in numeric_cols_to_normalize:
        if col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                # This case is for numeric columns that might be read as object type
                original_non_numeric_count = data[col].apply(lambda x: not isinstance(x, (int, float)) and pd.notna(x)).sum()
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if original_non_numeric_count > 0:
                    print(f"Warning: Column '{col}' (expected to be numeric) was converted to numeric. It originally contained {original_non_numeric_count} non-numeric string(s) which are now NaN(s).")
        else:
            print(f"Warning: Numeric feature '{col}' not found in data. It will not be processed or used.")

    # Drop rows with missing values (NaNs) in critical columns
    # Critical columns include the target and features that are essential for the model.
    # For 'defect', NaNs might arise from unexpected values.
    # For numeric_cols_to_normalize, NaNs might arise from 'coerce' or were originally present.
    # One-hot encoded columns don't introduce NaNs unless original column was all NaN (handled by dummy_na).

    # Identify columns that should not have NaNs for model training after this stage.
    # These typically include the target and all features that will be fed to the model.
    # For this example, let's assume all original numeric features and the defect column are critical.
    # One-hot encoded columns are derived, so their "missingness" is handled by dummy_na or absence of a category.

    critical_columns_for_dropna = ['defect'] + [col for col in numeric_cols_to_normalize if col in data.columns]

    # Report NaNs before dropping
    for col in critical_columns_for_dropna:
        if data[col].isnull().any():
            print(f"NaNs found in '{col}' before final dropna: {data[col].isnull().sum()} occurrences.")

    data = data.dropna(subset=critical_columns_for_dropna)
    print(f"Data after dropping rows with NaNs in critical columns: {data.shape[0]} rows remaining.")

    if data.empty:
        print("Warning: Data is empty after preprocessing and dropping NaNs. Further processing might fail.")
        return data

    # Normalize only the specified numeric features that still exist in the dataframe
    # Ensure these columns are indeed numeric before scaling
    final_numeric_features_to_scale = [col for col in numeric_cols_to_normalize if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]

    if not final_numeric_features_to_scale:
        print("Warning: No valid numeric features available for normalization after preprocessing.")
    else:
        print(f"Normalizing numeric features: {final_numeric_features_to_scale}")
        data[final_numeric_features_to_scale] = (
            data[final_numeric_features_to_scale] - data[final_numeric_features_to_scale].mean()
        ) / data[final_numeric_features_to_scale].std()
        # Check for NaNs again after normalization (e.g. if std is zero)
        if data[final_numeric_features_to_scale].isnull().any().any():
            print("Warning: NaNs introduced during normalization (e.g., std deviation is zero for a feature). Check data.")
            # Decide on handling: dropna again, impute, or error
            data = data.dropna(subset=final_numeric_features_to_scale)
            print(f"Data after dropping NaNs from normalization: {data.shape[0]} rows remaining.")


    return data

def train_and_evaluate(data):
    """
    Train and evaluate machine learning models.
    Returns a tuple: (best_model, all_metrics_dict).
    all_metrics_dict contains accuracy and classification report for each model.
    """
    if data.empty:
        print("Data is empty, skipping model training and evaluation.")
        return None, {}

    if 'defect' not in data.columns:
        print("Target column 'defect' not found in data. Skipping model training.")
        return None, {}

    y = data['defect']
    X = data.drop(columns=['defect'])

    if X.empty or y.empty or X.shape[1] == 0:
        print("Features (X) or target (y) is empty. Skipping model training.")
        return None, {}

    if len(y.unique()) < 2:
        print("Target variable 'defect' has less than 2 unique classes. Model training may fail or be meaningless.")
        # Depending on policy, either return or proceed with caution
        # return None, {}


    # Split data into training and testing sets
    # test_size adjusted if dataset is too small
    test_size = 0.3 if len(data) * 0.3 >= 2 else (1 / len(data) if len(data) >=2 else 0) # ensure at least 1 sample in test if possible
    if test_size == 0 and len(data) < 2 : # Cannot split if less than 2 samples
        print("Insufficient data to split into training and testing sets. Skipping model training.")
        return None, {}
    elif test_size == 0 and len(data) >=2 : # use one sample for test if data has at least 2 samples
        test_size = 1

    # stratify ensures similar class proportions in train and test if possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
    except ValueError as e:
        print(f"Error during train_test_split (possibly due to small sample size or class imbalance): {e}. Trying without stratification.")
        # Fallback if stratification fails (e.g. a class has only 1 member)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
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
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, zero_division=0)

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
            all_metrics[model_name] = {
                "accuracy": 0,
                "classification_report": "Error during processing."
            }

    if best_model:
        # Get the class name of the best model for a more dynamic message
        best_model_name = type(best_model).__name__
        print(f"Selected best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    else:
        print("No model was successfully trained or selected.")

    return best_model, all_metrics

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

if __name__ == "__main__":
    # File path to the welding dataset (replace with actual file path)
    file_path = 'welding_data.csv'

    # Load data
    welding_data = load_data(file_path)

    if welding_data is not None:
        # Preprocess data
        welding_data = preprocess_data(welding_data)

        if welding_data is not None and not welding_data.empty:
            # Visualize data
            # Note: visualize_data itself checks for empty data, but good practice to check here too.
            visualize_data(welding_data.copy()) # Pass a copy to avoid modification by reference issues if any

            # Train and evaluate models
            # Note: train_and_evaluate also checks for empty data.
            best_model, all_metrics = train_and_evaluate(welding_data.copy()) # Pass a copy

            if best_model:
                print(f"Best model ({type(best_model).__name__}) selected and ready for predictions.")
            else:
                print("No best model was selected. Check logs for details.")

            print("\nDetailed metrics for all models:")
            if all_metrics:
                for model_name, metrics in all_metrics.items():
                    print(f"--- {model_name} ---")
                    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                    print(f"  Classification Report:\n{metrics.get('classification_report', 'N/A')}")
            else:
                print("No metrics were generated.")

        elif welding_data is not None and welding_data.empty:
            print("Preprocessing resulted in empty data. Skipping visualization and model training.")
        else: # This case implies preprocess_data itself returned None (should not happen with current code)
            print("Preprocessing failed or returned None. Skipping visualization and model training.")

    else:
        print("Data loading failed. Please check the file path and data integrity.")

    print("\nScript execution finished.")
