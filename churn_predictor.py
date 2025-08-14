import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    """
    A comprehensive Customer Churn Prediction system using multiple ML algorithms.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.best_model = None
        self.best_model_name = ""
        self.best_score = 0
        
    def load_and_explore_data(self, file_path=None):
        """
        Load the dataset and perform initial exploration.
        If no file_path is provided, creates a synthetic dataset for demonstration.
        """
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                print("Dataset loaded successfully!")
            except FileNotFoundError:
                print("File not found. Creating synthetic dataset for demonstration...")
                self.df = self._create_synthetic_dataset()
        else:
            print("No dataset path provided. Creating synthetic dataset for demonstration...")
            self.df = self._create_synthetic_dataset()
            
        # Display basic information about the dataset
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nColumn names: {list(self.df.columns)}")
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        print(f"\nDataset info:")
        print(self.df.info())
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        return self.df
    
    def _create_synthetic_dataset(self):
        """
        Create a synthetic telecom churn dataset for demonstration purposes.
        """
        np.random.seed(42)
        n_samples = 5000
        
        # Generate synthetic data
        data = {
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'MonthlyCharges': np.round(np.random.uniform(20, 120, n_samples), 2),
            'TotalCharges': np.random.uniform(20, 8500, n_samples)
        }
        
        # Create churn based on some logical rules to make it realistic
        churn_prob = 0.1  # Base churn probability
        churn_indicators = []
        
        for i in range(n_samples):
            prob = churn_prob
            # Higher churn for month-to-month contracts
            if data['Contract'][i] == 'Month-to-month':
                prob += 0.15
            # Higher churn for senior citizens
            if data['SeniorCitizen'][i] == 1:
                prob += 0.1
            # Higher churn for high monthly charges
            if data['MonthlyCharges'][i] > 80:
                prob += 0.1
            # Lower churn for long tenure
            if data['tenure'][i] > 50:
                prob -= 0.15
            
            churn_indicators.append(1 if np.random.random() < prob else 0)
        
        data['Churn'] = churn_indicators
        
        # Add some missing values to demonstrate handling
        indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        for idx in indices:
            data['TotalCharges'][idx] = np.nan
        
        return pd.DataFrame(data)
    
    def preprocess_data(self):
        """
        Preprocess the data: handle missing values, encode categorical variables, scale features.
        """
        print("\n=== Data Preprocessing ===")
        
        # Make a copy of the dataframe
        df_processed = self.df.copy()
        
        # Handle missing values
        print("Handling missing values...")
        # For numerical columns, fill with median
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Churn' and df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        # Convert TotalCharges to numeric if it's object type (common in Kaggle dataset)
        if 'TotalCharges' in df_processed.columns and df_processed['TotalCharges'].dtype == 'object':
            df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
            df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
        
        # Convert target variable to binary
        if 'Churn' in df_processed.columns:
            if df_processed['Churn'].dtype == 'object':
                df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Churn']
        
        for col in categorical_cols:
            if df_processed[col].nunique() <= 10:  # Use label encoding for low cardinality
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            else:  # Use one-hot encoding for high cardinality
                df_encoded = pd.get_dummies(df_processed[col], prefix=col)
                df_processed = df_processed.drop(col, axis=1)
                df_processed = pd.concat([df_processed, df_encoded], axis=1)
        
        # Separate features and target
        if 'Churn' in df_processed.columns:
            self.X = df_processed.drop('Churn', axis=1)
            self.y = df_processed['Churn']
        else:
            print("Warning: 'Churn' column not found. Using last column as target.")
            self.X = df_processed.iloc[:, :-1]
            self.y = df_processed.iloc[:, -1]
        
        self.feature_columns = list(self.X.columns)
        
        # Scale features
        print("Scaling features...")
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=self.X.columns)
        
        print(f"Preprocessing completed!")
        print(f"Feature shape: {self.X_scaled.shape}")
        print(f"Target distribution:")
        print(self.y.value_counts(normalize=True))
        
        return self.X_scaled, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        """
        print(f"\n=== Splitting Data ===")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        print(f"Training set churn rate: {self.y_train.mean():.3f}")
        print(f"Testing set churn rate: {self.y_test.mean():.3f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple machine learning models.
        """
        print(f"\n=== Training Models ===")
        
        # Initialize models
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
        
        print("All models trained successfully!")
        return self.models
    
    def evaluate_models(self):
        """
        Evaluate all trained models and identify the best one.
        """
        print(f"\n=== Model Evaluation ===")
        
        results = {}
        plt.figure(figsize=(12, 8))
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Update best model
            if roc_auc > self.best_score:
                self.best_score = roc_auc
                self.best_model = model
                self.best_model_name = name
            
            # Print results
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Finalize ROC plot
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"\n=== Best Model: {self.best_model_name} (ROC-AUC: {self.best_score:.4f}) ===")
        
        # Display confusion matrix for best model
        best_predictions = results[self.best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, best_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        # Classification report for best model
        print(f"\nClassification Report - {self.best_model_name}:")
        print(classification_report(self.y_test, best_predictions))
        
        return results
    
    def predict_churn(self, customer_data):
        """
        Predict churn for a new customer.
        
        Args:
            customer_data: Dictionary with customer information
        """
        if self.best_model is None:
            print("No trained model available. Please train models first.")
            return None
        
        # Convert customer data to DataFrame
        customer_df = pd.DataFrame([customer_data])
        
        # Preprocess the customer data same way as training data
        for col in customer_df.columns:
            if col in self.label_encoders:
                # Handle unseen categories
                try:
                    customer_df[col] = self.label_encoders[col].transform(customer_df[col])
                except ValueError:
                    # If unseen category, use most common category
                    customer_df[col] = 0
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in customer_df.columns:
                customer_df[col] = 0
        
        # Reorder columns to match training data
        customer_df = customer_df[self.feature_columns]
        
        # Scale the features
        customer_scaled = self.scaler.transform(customer_df)
        
        # Make prediction
        churn_probability = self.best_model.predict_proba(customer_scaled)[0, 1]
        churn_prediction = self.best_model.predict(customer_scaled)[0]
        
        print(f"\n=== Churn Prediction ===")
        print(f"Customer will {'CHURN' if churn_prediction == 1 else 'NOT CHURN'}")
        print(f"Churn Probability: {churn_probability:.4f}")
        print(f"Model Used: {self.best_model_name}")
        
        return {
            'churn_prediction': churn_prediction,
            'churn_probability': churn_probability,
            'model_used': self.best_model_name
        }
    
    def save_model(self, filename='churn_model.pkl'):
        """
        Save the best model and preprocessing objects.
        """
        if self.best_model is None:
            print("No trained model available to save.")
            return
        
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'best_score': self.best_score
        }
        
        joblib.dump(model_package, filename)
        print(f"\nBest model ({self.best_model_name}) saved as {filename}")
        print(f"Model ROC-AUC Score: {self.best_score:.4f}")
    
    def load_model(self, filename='churn_model.pkl'):
        """
        Load a previously saved model.
        """
        try:
            model_package = joblib.load(filename)
            self.best_model = model_package['model']
            self.best_model_name = model_package['model_name']
            self.scaler = model_package['scaler']
            self.label_encoders = model_package['label_encoders']
            self.feature_columns = model_package['feature_columns']
            self.best_score = model_package['best_score']
            
            print(f"Model loaded successfully!")
            print(f"Model: {self.best_model_name}")
            print(f"ROC-AUC Score: {self.best_score:.4f}")
            
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found.")
            return False


def main():
    """
    Main function to demonstrate the complete churn prediction pipeline.
    """
    print("=== Customer Churn Prediction System ===")
    
    # Initialize the predictor
    predictor = ChurnPredictor()
    
    # Load data (using synthetic data for demonstration)
    # To use real data: predictor.load_and_explore_data('path/to/your/dataset.csv')
    predictor.load_and_explore_data()
    
    # Preprocess the data
    X, y = predictor.preprocess_data()
    
    # Split the data
    predictor.split_data()
    
    # Train models
    predictor.train_models()
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Save the best model
    predictor.save_model()
    
    # Example prediction for a new customer
    print("\n=== Example Customer Prediction ===")
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'InternetService': 'Fiber optic',
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0,
        'TotalCharges': 1020.0
    }
    
    prediction = predictor.predict_churn(sample_customer)
    
    # Interactive prediction (commented out for demo)
    # print("\n=== Interactive Prediction ===")
    # print("Enter customer details for churn prediction:")
    # # Add interactive input code here if needed
    
    print("\n=== Analysis Complete ===")
    print(f"Best Model: {predictor.best_model_name}")
    print(f"Best ROC-AUC Score: {predictor.best_score:.4f}")
    print(f"Model saved as: churn_model.pkl")


if __name__ == "__main__":
    main()