# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Function to load and preprocess the dataset
def load_and_preprocess_dataset(file_path):
    # Reading the dataset
    df = pd.read_csv(file_path)
    
    # Dynamically identify the target column (last column)
    target_column = df.columns[-1]
    print(f"Target column identified: {target_column}")
    
    # Display basic information about the dataset
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Shape:")
    print(df.shape)
    print("\nDataset Description:")
    print(df.describe())
    
    # Check for missing values
    df_missing_values = df.isnull().sum()
    print("\nMissing Values in Columns:")
    print(df_missing_values)
    
    # Analyze class distribution
    classes = df[target_column].value_counts()
    print("\nClass Distribution:")
    print(classes)
    
    # Plot class distribution
    sns.countplot(x=target_column, data=df)
    plt.title('Number of Fraud vs Non-Fraud Transactions')
    plt.show()
    
    return df, target_column

# Function to preprocess the data
def preprocess_data(df, target_column):
    # Drop unnecessary columns (e.g., 'Time' if it exists)
    if 'Time' in df.columns:
        df.drop('Time', axis=1, inplace=True)
        print("\nDropped 'Time' column.")
    
    # Handle date-time columns
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column contains strings
            try:
                # Try converting to datetime
                df[col] = pd.to_datetime(df[col])
                print(f"\nConverted '{col}' to datetime.")
                
                # Extract useful features from datetime (e.g., day, month, hour)
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_hour'] = df[col].dt.hour
                
                # Drop the original datetime column
                df.drop(col, axis=1, inplace=True)
                print(f"Dropped original datetime column '{col}'.")
            except:
                # If conversion fails, assume it's a categorical column
                print(f"\nColumn '{col}' is not a datetime column. Treating as categorical.")
    
    # Separate fraud and non-fraud data
    data_fraud = df[df[target_column] == 1]
    data_non_fraud = df[df[target_column] == 0]
    
    # Dynamically identify numerical columns (excluding the target column)
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_columns.remove(target_column)  # Remove the target column
    
    # Plot distribution of numerical columns
    for col in numerical_columns:
        plt.figure(figsize=(8, 5))
        ax = sns.distplot(data_fraud[col], label='fraudulent', hist=False)
        ax = sns.distplot(data_non_fraud[col], label='non-fraudulent', hist=False)
        ax.set(xlabel=col)
        plt.legend()
        plt.show()
    
    # Split data into features (X) and target (y)
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)
    
    # Scale numerical columns
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    return X_train, X_test, y_train, y_test

# Function to evaluate and display results
def evaluate_model(model_name, model, X_test, y_test):
    # Prediction on the test set
    y_test_pred = model.predict(X_test)
    
    # Confusion matrix
    print(f"\n------------------ {model_name} Confusion Matrix --------------------")
    c_matrix = metrics.confusion_matrix(y_test, y_test_pred)
    print(c_matrix)
    
    cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()
    
    # Classification report
    print(f"\n------------------ {model_name} Classification Report --------------------")
    print(classification_report(y_test, y_test_pred))
    
    # ROC Curve
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    draw_roc(y_test, y_test_pred_proba)
    
    # Store results
    results.loc[len(results)] = [
        model_name,
        metrics.accuracy_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred),
        metrics.roc_auc_score(y_test, y_test_pred_proba)
    ]

# Function to draw ROC curve
def draw_roc(actual, probs):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess the dataset
    file_path = 'archive/fraudTest.csv'  # Replace with user-provided file path
    df, target_column = load_and_preprocess_dataset(file_path)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
    
    # Initialize results dataframe
    results = pd.DataFrame(columns=['Model Name', 'Accuracy', 'F1-score', 'ROC'])
    
    # Train and evaluate models
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # Logistic Regression
    logistic_model = LogisticRegression(C=0.01)
    logistic_model.fit(X_train, y_train)
    evaluate_model("Logistic Regression", logistic_model, X_test, y_test)
    
    # XGBoost
    xgb_model = XGBClassifier(learning_rate=0.2, max_depth=2, n_estimators=200, subsample=0.9, objective='binary:logistic')
    xgb_model.fit(X_train, y_train)
    evaluate_model("XGBoost", xgb_model, X_test, y_test)
    
    # Decision Tree
    decision_tree_model = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=100, min_samples_split=100, random_state=100)
    decision_tree_model.fit(X_train, y_train)
    evaluate_model("Decision Tree", decision_tree_model, X_test, y_test)
    
    # Random Forest
    random_forest_model = RandomForestClassifier(bootstrap=True, max_depth=5, min_samples_leaf=50, min_samples_split=50, max_features=10, n_estimators=100)
    random_forest_model.fit(X_train, y_train)
    evaluate_model("Random Forest", random_forest_model, X_test, y_test)
    
    # Display final results
    print("\nFinal Results:")
    print(results.sort_values(by="ROC", ascending=False))