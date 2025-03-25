"""
Modeling pipeline stages for email classification
"""
import logging
import time
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Any, Optional, Union

# Add the root directory to the Python path to allow importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_pipeline import PipelineStage, PipelineContext, pipeline_stage
from .data_wrapper import Data
from Config import Config
from model.randomforest import RandomForest

logger = logging.getLogger(__name__)

@pipeline_stage(
    requires=["embeddings", "preprocessed_data"],
    provides=["data", "grouped_data"]
)
class DataPreparationStage(PipelineStage):
    """Stage to prepare data for modeling"""
    
    def __init__(self, target_col: str = None, name: str = "DataPreparationStage"):
        super().__init__(name)
        self.target_col = target_col or Config.CLASS_COL
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Prepare data for modeling by grouping and creating Data objects"""
        df = context.get("preprocessed_data")
        X = context.get("embeddings")
        logger.info(f"Preparing data for modeling, target column: {self.target_col}")
        
        # Create a default data object for the entire dataset
        logger.info(f"Creating data object for target column: {self.target_col}")
        data_wrapper = Data(X, df, target_col=self.target_col)
        
        # Group by Type 1 (if available)
        if Config.GROUPED in df.columns:
            grouped_data = {}
            grouped_df = df.groupby(Config.GROUPED)
            
            for name, group_df in grouped_df:
                logger.info(f"Creating group data for {Config.GROUPED}={name}")
                
                # Get embeddings for this group
                mask = df[Config.GROUPED] == name
                group_X = X[mask.values]
                
                # Create Data for this group
                group_data = Data(group_X, group_df, target_col=self.target_col)
                
                if group_data.is_valid:
                    grouped_data[name] = group_data
                else:
                    logger.warning(f"Skipping invalid group: {name}")
            
            context.set("grouped_data", grouped_data)
            logger.info(f"Created data objects for {len(grouped_data)} groups")
        
        context.set("data", data_wrapper)
        return context


@pipeline_stage(
    requires=["grouped_data"],
    provides=["single_models"]
)
class SingleModelStage(PipelineStage):
    """Stage to train standard single target models"""
    
    def __init__(self, n_estimators: int = 500, name: str = "SingleModelStage"):
        super().__init__(name)
        self.n_estimators = n_estimators
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Train standard single target models for each group"""
        grouped_data = context.get("grouped_data")
        models = {}
        
        logger.info(f"Training single target models for {len(grouped_data)} groups")
        
        for group_name, data in grouped_data.items():
            logger.info(f"Training model for group: {group_name}")
            
            if not data.is_valid:
                logger.warning(f"Skipping invalid data for group: {group_name}")
                continue
                
            start_time = time.time()
            
            # Create and train the model
            model = RandomForest(
                f"SingleModel_{group_name}",
                data.get_embeddings(),
                data.get_type(),
                mode="single",
                n_estimators=self.n_estimators
            )
            
            model.train(data)
            model.predict(data.get_X_test())
            
            # Store evaluation metrics
            y_test = data.get_type_y_test()
            predictions = model.predictions
            
            # Store the model
            models[group_name] = {
                "model": model,
                "metrics": model.metrics,
                "training_time": time.time() - start_time
            }
            
            logger.info(f"Completed training for group {group_name} in {models[group_name]['training_time']:.2f} seconds")
        
        context.set("single_models", models)
        return context


@pipeline_stage(
    requires=["grouped_data"],
    provides=["chained_models"]
)
class ChainedModelStage(PipelineStage):
    """Stage to train chained multi-output models (Design Decision 1)"""
    
    def __init__(self, n_estimators: int = 300, name: str = "ChainedModelStage"):
        super().__init__(name)
        self.n_estimators = n_estimators
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Train chained multi-output models for each group"""
        grouped_data = context.get("grouped_data")
        models = {}
        
        logger.info(f"Training chained multi-output models for {len(grouped_data)} groups")
        
        for group_name, data in grouped_data.items():
            logger.info(f"Training chained model for group: {group_name}")
            
            if not data.is_valid:
                logger.warning(f"Skipping invalid data for group: {group_name}")
                continue
                
            start_time = time.time()
            
            # Create and train the chained model
            model = RandomForest(
                f"ChainedModel_{group_name}",
                data.get_embeddings(),
                data.get_type(),
                mode="chained",
                n_estimators=self.n_estimators
            )
            
            model.train(data)
            model.predict(data.get_X_test(), data)
            
            # Store the model
            models[group_name] = {
                "model": model,
                "metrics": model.metrics,
                "training_time": time.time() - start_time
            }
            
            # Save the model
            model.save_model()
            
            logger.info(f"Completed chained model training for group {group_name} in {models[group_name]['training_time']:.2f} seconds")
        
        context.set("chained_models", models)
        return context


@pipeline_stage(
    requires=["grouped_data"],
    provides=["hierarchical_models"]
)
class HierarchicalModelStage(PipelineStage):
    """Stage to train hierarchical models (Design Decision 2)"""
    
    def __init__(self, n_estimators: int = 500, child_estimators: int = 200, name: str = "HierarchicalModelStage"):
        super().__init__(name)
        self.n_estimators = n_estimators
        self.child_estimators = child_estimators
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Train hierarchical models for each group"""
        grouped_data = context.get("grouped_data")
        models = {}
        
        logger.info(f"Training hierarchical models for {len(grouped_data)} groups")
        
        for group_name, data in grouped_data.items():
            logger.info(f"Training hierarchical model for group: {group_name}")
            
            if not data.is_valid:
                logger.warning(f"Skipping invalid data for group: {group_name}")
                continue
                
            start_time = time.time()
            model_creation_time = 0
            
            # Get data properties
            target_col = data.get_target_column()
            
            # Create the base model for Type 2
            logger.info(f"Training base model for {target_col}")
            base_model = RandomForest(
                f"HierarchicalModel_{group_name}",
                data.get_embeddings(),
                data.get_type(),
                mode="hierarchical",
                n_estimators=self.n_estimators
            )
            
            # Train the base model
            base_model.train(data)
            base_model.predict(data.get_X_test(), data)
            
            # Save test data for evaluation
            base_model.test_data = data
            base_model.test_features = data.get_X_test()
            base_model.test_targets = data.get_type_y_test()
            
            # If this is Type 2, create child models for each Type 2 class
            if target_col == Config.TYPE_COLS[0] and hasattr(data, 'classes'):
                original_df = data.get_original_df()
                embeddings = data.get_embeddings()
                
                # All models for this group
                all_models = {target_col: base_model}
                
                # For each unique class in Type 2
                for class_value in data.get_classes():
                    # Skip empty classes
                    if class_value is None or (isinstance(class_value, str) and not class_value.strip()):
                        continue
                        
                    # Create filter condition for this class
                    filter_condition = {Config.TYPE_COLS[0]: class_value}
                    logger.info(f"Training Type 3 model for {Config.TYPE_COLS[0]}={class_value}")
                    
                    child_start_time = time.time()
                    
                    # Create data for Type 3 model with this Type 2 class
                    type3_data = Data(
                        embeddings,
                        original_df,
                        target_col=Config.TYPE_COLS[1],
                        filter_condition=filter_condition,
                        mode="hierarchical"
                    )
                    
                    # Only continue if we have valid data and classes
                    if type3_data.is_valid and hasattr(type3_data, 'classes') and len(type3_data.get_classes()) > 0:
                        # Train Type 3 model
                        type3_model = RandomForest(
                            f"Type3Model_{class_value}",
                            type3_data.get_embeddings(),
                            type3_data.get_type(),
                            mode="hierarchical",
                            n_estimators=self.child_estimators
                        )
                        
                        type3_model.train(type3_data)
                        type3_model.predict(type3_data.get_X_test(), type3_data)
                        
                        # Save test data for evaluation
                        type3_model.test_data = type3_data
                        type3_model.test_features = type3_data.get_X_test()
                        type3_model.test_targets = type3_data.get_type_y_test()
                        
                        # Add as child model
                        base_model.add_child_model(type3_model)
                        
                        # Store in all_models dictionary
                        model_key = f"{Config.TYPE_COLS[0]}={class_value}_{Config.TYPE_COLS[1]}"
                        all_models[model_key] = type3_model
                        
                        # For each Type 3 class, create Type 4 models
                        for type3_class in type3_data.get_classes():
                            # Skip empty classes
                            if type3_class is None or (isinstance(type3_class, str) and not type3_class.strip()):
                                continue
                                
                            # Create filter condition for this Type 3 class
                            type4_filter = {
                                Config.TYPE_COLS[0]: class_value, 
                                Config.TYPE_COLS[1]: type3_class
                            }
                            
                            logger.info(f"Training Type 4 model for {Config.TYPE_COLS[0]}={class_value}, {Config.TYPE_COLS[1]}={type3_class}")
                            
                            # Create data for Type 4 model
                            type4_data = Data(
                                embeddings,
                                original_df,
                                target_col=Config.TYPE_COLS[2],
                                filter_condition=type4_filter,
                                mode="hierarchical"
                            )
                            
                            if type4_data.is_valid and hasattr(type4_data, 'classes') and len(type4_data.get_classes()) > 0:
                                # Train Type 4 model
                                type4_model = RandomForest(
                                    f"Type4Model_{class_value}_{type3_class}",
                                    type4_data.get_embeddings(),
                                    type4_data.get_type(),
                                    mode="hierarchical",
                                    n_estimators=self.child_estimators
                                )
                                
                                type4_model.train(type4_data)
                                type4_model.predict(type4_data.get_X_test(), type4_data)
                                
                                # Save test data for evaluation
                                type4_model.test_data = type4_data
                                type4_model.test_features = type4_data.get_X_test()
                                type4_model.test_targets = type4_data.get_type_y_test()
                                
                                # Add as child model
                                type3_model.add_child_model(type4_model)
                    
                    model_creation_time += time.time() - child_start_time
                
                # Save the hierarchical model (which contains all child models)
                base_model.save_model()
                
                # Store all models for this group
                models[group_name] = {
                    "base_model": base_model,
                    "all_models": all_models,
                    "metrics": base_model.metrics,
                    "total_time": time.time() - start_time,
                    "model_creation_time": model_creation_time
                }
                
                logger.info(f"Completed hierarchical model training for group {group_name} in {models[group_name]['total_time']:.2f} seconds")
        
        context.set("hierarchical_models", models)
        return context 