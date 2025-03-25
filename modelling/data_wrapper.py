"""
Data wrapper for pipeline architecture
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any

# Add the root directory to the Python path to allow importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Config import Config

logger = logging.getLogger(__name__)

class Data:
    """Adapts the functionality of the original Data class for pipeline architecture"""
    
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame,
                 target_col: str = None,
                 filter_condition: Dict[str, Any] = None,
                 min_class_samples: int = None,
                 test_size: float = 0.2,
                 random_seed: int = None,
                 mode: str = "standard") -> None:
        """
        Initialize the data wrapper
        
        Args:
            X: The feature embeddings array
            df: The dataframe with labels
            target_col: The target column to use (default from Config)
            filter_condition: Filter condition for hierarchical modeling
            min_class_samples: Minimum samples per class (default from Config)
            test_size: Test split size (default 0.2)
            random_seed: Random seed for reproducibility (default from Config)
            mode: The modeling mode ("standard", "chained", or "hierarchical")
        """
        # Set configuration parameters
        self.target_col = target_col or Config.CLASS_COL
        self.filter_condition = filter_condition
        self.mode = mode
        
        # For hierarchical modeling with filtered data, use a lower threshold
        if mode == "hierarchical" and filter_condition:
            self.min_class_samples = min_class_samples or min(Config.MIN_CLASS_SAMPLES, 2)
        else:
            self.min_class_samples = min_class_samples or Config.MIN_CLASS_SAMPLES
            
        self.random_seed = random_seed or Config.RANDOM_SEED
        self.test_size = test_size
        
        # Store original data
        self.original_df = df.copy()
        self.embeddings = X
        self.is_valid = True
        
        # Ensure X attribute is maintained for test data
        self.X = X
        
        # Log initialization info
        logger.info(f"Initializing Data for target column: {self.target_col}")
        if filter_condition:
            logger.info(f"With filter condition: {filter_condition}")
        
        # Process and prepare the data
        self._prepare_data(df, X)
    
    def _prepare_data(self, df: pd.DataFrame, X: np.ndarray) -> None:
        """Prepare the data for modeling"""
        # Check if target column exists
        if self.target_col not in df.columns:
            logger.error(f"Target column {self.target_col} not found in DataFrame. Available columns: {df.columns.tolist()}")
            self.is_valid = False
            return
        
        # Apply filter condition if provided
        filtered_df = df
        filtered_X = X
        if self.filter_condition:
            mask = pd.Series(True, index=df.index)
            for col, val in self.filter_condition.items():
                mask &= (df[col] == val)
                
            if not mask.any():
                logger.warning(f"No data found for filter condition: {self.filter_condition}")
                self.is_valid = False
                return
                
            filtered_df = df.loc[mask]
            filtered_X = X[mask.values]
            logger.info(f"Applied filter condition, resulting in {len(filtered_df)} rows")
        
        # Get target variable
        self.y = filtered_df[self.target_col].to_numpy()
        y_series = pd.Series(self.y)
        
        # Prepare chained targets if needed
        if Config.CHAINED_MODE and self.target_col == Config.TYPE_COLS[0]:
            self._prepare_chained_targets(filtered_df)
        
        # Filter by minimum class samples
        class_counts = y_series.value_counts()
        good_y_value = class_counts[class_counts >= self.min_class_samples].index
        
        if len(good_y_value) < 1:
            logger.warning(f"None of the classes have more than {self.min_class_samples} records. Skipping...")
            self.is_valid = False
            return
            
        # Create mask for good values
        mask = y_series.isin(good_y_value)
        self.y = self.y[mask]
        X_good = filtered_X[mask]
        
        # Calculate appropriate test size
        new_test_size = min(self.test_size, self.test_size * len(filtered_X) / max(1, len(X_good)))
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, self.y, test_size=new_test_size, 
            random_state=self.random_seed, stratify=self.y
        )
        
        # For chained modeling, also split the combined targets
        if Config.CHAINED_MODE and self.target_col == Config.TYPE_COLS[0]:
            self._split_chained_targets(mask, X_good, new_test_size)
        
        self.classes = good_y_value.tolist()
        logger.info(f"Created data splits: train={len(self.X_train)}, test={len(self.X_test)}")
        logger.info(f"Class distribution: {', '.join([f'{c}: {sum(self.y_train == c)}' for c in self.classes])}")

    def _prepare_chained_targets(self, filtered_df: pd.DataFrame) -> None:
        """Prepare the combined targets for chained modeling"""
        self.y2 = filtered_df[Config.TYPE_COLS[0]].to_numpy()
        
        if Config.TYPE_COLS[1] in filtered_df.columns:
            self.y3 = filtered_df[Config.TYPE_COLS[1]].to_numpy()
            # Create combined target for Type 2 + Type 3
            self.y2_y3 = np.array([f"{y2}+{y3}" for y2, y3 in zip(self.y2, self.y3)])
        else:
            self.y3 = None
            self.y2_y3 = None
            logger.warning(f"Type 3 column ({Config.TYPE_COLS[1]}) not found for chained targets")
            
        if Config.TYPE_COLS[1] in filtered_df.columns and Config.TYPE_COLS[2] in filtered_df.columns:
            self.y4 = filtered_df[Config.TYPE_COLS[2]].to_numpy()
            # Create combined target for Type 2 + Type 3 + Type 4
            self.y2_y3_y4 = np.array([f"{y2}+{y3}+{y4}" for y2, y3, y4 in zip(self.y2, self.y3, self.y4)])
        else:
            self.y4 = None
            self.y2_y3_y4 = None
            logger.warning(f"Type 4 column ({Config.TYPE_COLS[2]}) not found for chained targets")
            
    def _split_chained_targets(self, mask: pd.Series, X_good: np.ndarray, new_test_size: float) -> None:
        """Split the combined targets for chained modeling"""
        if hasattr(self, 'y2_y3') and self.y2_y3 is not None:
            y2_y3_good = self.y2_y3[mask]
            _, _, self.y2_y3_train, self.y2_y3_test = train_test_split(
                X_good, y2_y3_good, test_size=new_test_size, 
                random_state=self.random_seed, stratify=self.y
            )
            logger.info(f"Split y2_y3 target: {len(self.y2_y3_train)} train, {len(self.y2_y3_test)} test")
        
        if hasattr(self, 'y2_y3_y4') and self.y2_y3_y4 is not None:
            y2_y3_y4_good = self.y2_y3_y4[mask]
            _, _, self.y2_y3_y4_train, self.y2_y3_y4_test = train_test_split(
                X_good, y2_y3_y4_good, test_size=new_test_size, 
                random_state=self.random_seed, stratify=self.y
            )
            logger.info(f"Split y2_y3_y4 target: {len(self.y2_y3_y4_train)} train, {len(self.y2_y3_y4_test)} test")
    
    # Interface getters for compatibility with original code
    def get_type(self) -> np.ndarray:
        """Get the target variable"""
        return self.y
        
    def get_X_train(self) -> np.ndarray:
        """Get the training features"""
        return self.X_train
        
    def get_X_test(self) -> np.ndarray:
        """Get the test features"""
        return self.X_test
        
    def get_type_y_train(self) -> np.ndarray:
        """Get the training target"""
        return self.y_train
        
    def get_type_y_test(self) -> np.ndarray:
        """Get the test target"""
        return self.y_test
        
    def get_embeddings(self) -> np.ndarray:
        """Get all embeddings"""
        return self.embeddings
        
    def get_original_df(self) -> pd.DataFrame:
        """Get the original dataframe"""
        return self.original_df
        
    def get_target_column(self) -> str:
        """Get the target column name"""
        return self.target_col
        
    def get_filter_condition(self) -> Dict[str, Any]:
        """Get the filter condition"""
        return self.filter_condition
        
    # Methods for chained modeling
    def get_y2_y3_train(self) -> Optional[np.ndarray]:
        """Get the combined Type 2 + Type 3 training target"""
        return self.y2_y3_train if hasattr(self, 'y2_y3_train') else None
        
    def get_y2_y3_test(self) -> Optional[np.ndarray]:
        """Get the combined Type 2 + Type 3 test target"""
        return self.y2_y3_test if hasattr(self, 'y2_y3_test') else None
        
    def get_y2_y3_y4_train(self) -> Optional[np.ndarray]:
        """Get the combined Type 2 + Type 3 + Type 4 training target"""
        return self.y2_y3_y4_train if hasattr(self, 'y2_y3_y4_train') else None
        
    def get_y2_y3_y4_test(self) -> Optional[np.ndarray]:
        """Get the combined Type 2 + Type 3 + Type 4 test target"""
        return self.y2_y3_y4_test if hasattr(self, 'y2_y3_y4_test') else None
    
    def get_classes(self) -> List[Any]:
        """Get the unique class values"""
        return self.classes if hasattr(self, 'classes') else [] 