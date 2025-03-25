import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random

# Use Config for consistent seed
seed = Config.RANDOM_SEED
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame,
                 target_col: str = None,
                 filter_condition: dict = None) -> None:

        # Use provided target column or default from Config
        self.target_col = target_col or Config.CLASS_COL
        self.filter_condition = filter_condition
        self.original_df = df.copy()
        self.embeddings = X
        
        # Early return conditions
        if self.target_col not in df.columns:
            print(f"Target column {self.target_col} not found in DataFrame. Available columns: {df.columns.tolist()}")
            self.X_train = None
            return
        
        # Filter data if filter condition is provided (for hierarchical modeling)
        filtered_df = df
        filtered_X = X
        if filter_condition:
            mask = pd.Series(True, index=df.index)
            for col, val in filter_condition.items():
                mask &= (df[col] == val)
                
            if not mask.any():
                print(f"No data found for filter condition: {filter_condition}")
                self.X_train = None
                return
                
            filtered_df = df.loc[mask]
            filtered_X = X[mask.values]
        
        # Get target variable
        y = filtered_df[self.target_col].to_numpy()
        y_series = pd.Series(y)
        
        # Prepare combined targets for chained modeling (only when needed)
        if Config.CHAINED_MODE and self.target_col == Config.TYPE_COLS[0]:
            self._prepare_chained_targets(filtered_df)
        
        # Filter by minimum class samples
        class_counts = y_series.value_counts()
        good_y_value = class_counts[class_counts >= Config.MIN_CLASS_SAMPLES].index
        
        if len(good_y_value) < 1:
            print(f"None of the classes have more than {Config.MIN_CLASS_SAMPLES} records: Skipping...")
            self.X_train = None
            return
            
        # Create mask for good values
        mask = y_series.isin(good_y_value)
        self.y = y[mask]
        X_good = filtered_X[mask]
        
        # Calculate appropriate test size
        new_test_size = min(0.2, 0.2 * len(filtered_X) / max(1, len(X_good)))
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, self.y, test_size=new_test_size, random_state=seed, stratify=self.y
        )
        
        # For chained modeling, also split the combined targets
        if Config.CHAINED_MODE and self.target_col == Config.TYPE_COLS[0]:
            self._split_chained_targets(mask, X_good, new_test_size)
        
        self.classes = good_y_value.tolist()

    def _prepare_chained_targets(self, filtered_df):
        """Prepare the combined targets for chained modeling"""
        self.y2 = filtered_df[Config.TYPE_COLS[0]].to_numpy()
        
        if Config.TYPE_COLS[1] in filtered_df.columns:
            self.y3 = filtered_df[Config.TYPE_COLS[1]].to_numpy()
            self.y2_y3 = np.array([f"{y2}+{y3}" for y2, y3 in zip(self.y2, self.y3)])
        else:
            self.y3 = None
            self.y2_y3 = None
            
        if Config.TYPE_COLS[1] in filtered_df.columns and Config.TYPE_COLS[2] in filtered_df.columns:
            self.y4 = filtered_df[Config.TYPE_COLS[2]].to_numpy()
            self.y2_y3_y4 = np.array([f"{y2}+{y3}+{y4}" for y2, y3, y4 in zip(self.y2, self.y3, self.y4)])
        else:
            self.y4 = None
            self.y2_y3_y4 = None
            
    def _split_chained_targets(self, mask, X_good, new_test_size):
        """Split the combined targets for chained modeling"""
        if hasattr(self, 'y2_y3') and self.y2_y3 is not None:
            y2_y3_good = self.y2_y3[mask]
            _, _, self.y2_y3_train, self.y2_y3_test = train_test_split(
                X_good, y2_y3_good, test_size=new_test_size, random_state=seed, stratify=self.y
            )
        
        if hasattr(self, 'y2_y3_y4') and self.y2_y3_y4 is not None:
            y2_y3_y4_good = self.y2_y3_y4[mask]
            _, _, self.y2_y3_y4_train, self.y2_y3_y4_test = train_test_split(
                X_good, y2_y3_y4_good, test_size=new_test_size, random_state=seed, stratify=self.y
            )
            
    # Getters - Use property decorators for cleaner access
    @property
    def type(self):
        return self.y
        
    def get_type(self):
        return self.y
        
    def get_X_train(self):
        return self.X_train
        
    def get_X_test(self):
        return self.X_test
        
    def get_type_y_train(self):
        return self.y_train
        
    def get_type_y_test(self):
        return self.y_test
        
    def get_embeddings(self):
        return self.embeddings
        
    def get_original_df(self):
        return self.original_df
        
    def get_target_column(self):
        return self.target_col
        
    def get_filter_condition(self):
        return self.filter_condition
        
    # Methods for chained modeling
    def get_y2_y3_train(self):
        return self.y2_y3_train if hasattr(self, 'y2_y3_train') else None
        
    def get_y2_y3_test(self):
        return self.y2_y3_test if hasattr(self, 'y2_y3_test') else None
        
    def get_y2_y3_y4_train(self):
        return self.y2_y3_y4_train if hasattr(self, 'y2_y3_y4_train') else None
        
    def get_y2_y3_y4_test(self):
        return self.y2_y3_y4_test if hasattr(self, 'y2_y3_y4_test') else None

