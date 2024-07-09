import pandas as pd
from src.data_preprocessing import preprocess_data, split_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

def load_data(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    # Load data
    data_path = './data/credit_risk_data.csv'
    data = load_data(data_path)

    # Preprocess data
    data = preprocess_data(data)

    # Ensure the target column is categorical
    target_column = 'loan_status'  # Replace with your actual target column name
    data[target_column] = data[target_column].astype('int')

    # Split data into features and target
    X_train, X_test, y_train, y_test = split_data(data, target_column)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    report, roc_auc = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Report:")
    print(report)
    print(f"ROC AUC Score: {roc_auc}")
