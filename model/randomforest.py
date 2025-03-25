import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from numpy import *
import random
from Config import Config
import os

seed = Config.RANDOM_SEED
# Set random seeds for reproducibility
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray,
                 mode: str = "single",
                 n_estimators: int = 500,
                 n_jobs: int = -1) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mode = mode  # "single", "chained", or "hierarchical"
        
        # Use fewer trees for faster training without sacrificing much accuracy
        # n_jobs=-1 uses all available cores for parallel processing
        self.mdl = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=seed, 
            class_weight='balanced_subsample',
            n_jobs=n_jobs,
            # Add additional parameters for potentially better performance
            bootstrap=True,
            max_features='sqrt',
            oob_score=True,
            warm_start=False  # Set to True for incremental training
        )
        
        self.predictions = None
        self.data_transform()
        
        # For chained and hierarchical models
        self.models = {}
        self.child_models = []
        self.metrics = {}

    def train(self, data) -> None:
        """Train the model(s) based on mode"""
        if self.mode == "single" or self.mode == "hierarchical":
            # Standard training for single target or hierarchical base models
            self.mdl = self.mdl.fit(data.get_X_train(), data.get_type_y_train())
        elif self.mode == "chained":
            # For chained mode, we train multiple models
            # Model for Type 2
            self.mdl = self.mdl.fit(data.get_X_train(), data.get_type_y_train())
            
            # Model for Type 2 + Type 3 (if available)
            if data.get_y2_y3_train() is not None:
                self.models['y2_y3'] = RandomForestClassifier(
                    n_estimators=self.mdl.n_estimators, 
                    random_state=seed, 
                    class_weight='balanced_subsample',
                    n_jobs=-1,
                    bootstrap=True,
                    max_features='sqrt'
                ).fit(data.get_X_train(), data.get_y2_y3_train())
            
            # Model for Type 2 + Type 3 + Type 4 (if available)
            if data.get_y2_y3_y4_train() is not None:
                self.models['y2_y3_y4'] = RandomForestClassifier(
                    n_estimators=self.mdl.n_estimators, 
                    random_state=seed, 
                    class_weight='balanced_subsample',
                    n_jobs=-1,
                    bootstrap=True,
                    max_features='sqrt'
                ).fit(data.get_X_train(), data.get_y2_y3_y4_train())
                
        # Save feature importance
        if hasattr(self.mdl, 'feature_importances_'):
            self.feature_importances_ = self.mdl.feature_importances_

    def predict(self, X_test: pd.Series, data=None):
        """Make predictions based on mode"""
        # Store test data for evaluation
        self.test_features = X_test
        if data is not None:
            if self.mode == "single" or self.mode == "hierarchical":
                self.test_targets = data.get_type_y_test()
                self.test_data = data
            elif self.mode == "chained":
                self.test_targets = np.column_stack([
                    data.get_type_y_test(),
                    data.get_y2_y3_test() if data.get_y2_y3_test() is not None else np.full(len(data.get_type_y_test()), ""),
                    data.get_y2_y3_y4_test() if data.get_y2_y3_y4_test() is not None else np.full(len(data.get_type_y_test()), "")
                ])
                self.test_data = data
        
        if self.mode == "single" or self.mode == "hierarchical":
            # Standard prediction
            self.predictions = self.mdl.predict(X_test)
        elif self.mode == "chained":
            # Predict all target combinations
            self.predictions = self.mdl.predict(X_test)
            
            if 'y2_y3' in self.models:
                self.predictions_y2_y3 = self.models['y2_y3'].predict(X_test)
            
            if 'y2_y3_y4' in self.models:
                self.predictions_y2_y3_y4 = self.models['y2_y3_y4'].predict(X_test)
        
        return self.predictions

    def print_results(self, data):
        """Print classification reports based on mode"""
        if self.mode == "single" or self.mode == "hierarchical":
            # Calculate and store metrics
            y_test = data.get_type_y_test()
            self.metrics = {
                'accuracy': accuracy_score(y_test, self.predictions),
                'f1_macro': f1_score(y_test, self.predictions, average='macro'),
                'f1_weighted': f1_score(y_test, self.predictions, average='weighted')
            }
            
            # Print results for single target
            print(f"Results for {data.get_target_column()}:")
            print(classification_report(y_test, self.predictions))
            
        elif self.mode == "chained":
            # Print results for all target combinations
            y_test = data.get_type_y_test()
            self.metrics = {
                'accuracy_type2': accuracy_score(y_test, self.predictions),
                'f1_macro_type2': f1_score(y_test, self.predictions, average='macro'),
                'f1_weighted_type2': f1_score(y_test, self.predictions, average='weighted')
            }
            
            print("Results for Type 2:")
            print(classification_report(y_test, self.predictions))
            
            if hasattr(self, 'predictions_y2_y3') and data.get_y2_y3_test() is not None:
                y2_y3_test = data.get_y2_y3_test()
                self.metrics.update({
                    'accuracy_type2_y3': accuracy_score(y2_y3_test, self.predictions_y2_y3),
                    'f1_macro_type2_y3': f1_score(y2_y3_test, self.predictions_y2_y3, average='macro'),
                    'f1_weighted_type2_y3': f1_score(y2_y3_test, self.predictions_y2_y3, average='weighted')
                })
                print("\nResults for Type 2 + Type 3:")
                print(classification_report(y2_y3_test, self.predictions_y2_y3))
            
            if hasattr(self, 'predictions_y2_y3_y4') and data.get_y2_y3_y4_test() is not None:
                y2_y3_y4_test = data.get_y2_y3_y4_test()
                self.metrics.update({
                    'accuracy_type2_y3_y4': accuracy_score(y2_y3_y4_test, self.predictions_y2_y3_y4),
                    'f1_macro_type2_y3_y4': f1_score(y2_y3_y4_test, self.predictions_y2_y3_y4, average='macro'),
                    'f1_weighted_type2_y3_y4': f1_score(y2_y3_y4_test, self.predictions_y2_y3_y4, average='weighted')
                })
                print("\nResults for Type 2 + Type 3 + Type 4:")
                print(classification_report(y2_y3_y4_test, self.predictions_y2_y3_y4))
    
    def add_child_model(self, model):
        """Add a child model for hierarchical modeling"""
        self.child_models.append(model)
    
    def get_predictions(self):
        """Get predictions for current model"""
        return self.predictions
    
    def get_metrics(self):
        """Get evaluation metrics"""
        return self.metrics
    
    def save_model(self, path="./saved_models"):
        """Save the trained model to disk"""
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self, f"{path}/{self.model_name}.pkl")
        print(f"Model saved to {path}/{self.model_name}.pkl")
        
    @staticmethod
    def load_model(model_path):
        """Load a trained model from disk"""
        return joblib.load(model_path)
        
    def data_transform(self) -> None:
        """Any data transformations needed before training"""
        pass 