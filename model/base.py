from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any, List, Optional, Union


class BaseModel(ABC):
    """Base model class for all models"""
    
    def __init__(self):
        """Initialize the base model"""
        self.metrics = {}
        self.feature_importances_ = None
        self.test_data = None
        self.test_features = None
        self.test_targets = None
        self.predictions = None


    @abstractmethod
    def train(self, data) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X_test: pd.Series) -> np.ndarray:
        """Make predictions using the model"""
        pass

    @abstractmethod
    def data_transform(self) -> None:
        """
        Transform data if needed before training
        """
        return

    def build(self, values={}):
        """
        Build the model with given parameters
        """
        self.__dict__.update(values)
        return self 