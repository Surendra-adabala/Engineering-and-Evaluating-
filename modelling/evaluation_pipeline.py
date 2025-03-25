"""
Evaluation pipeline stages for email classification
"""
import logging
import time
import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Any, Optional, Union
import re
from datetime import datetime

# Add the root directory to the Python path to allow importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_pipeline import PipelineStage, PipelineContext, pipeline_stage
from Config import Config
from .data_wrapper import Data
from model.randomforest import RandomForest
from utils.metrics_utils import calculate_multiclass_metrics, calculate_multilabel_metrics
from utils.visualization_utils import (
    create_confusion_matrix,
    plot_feature_importance,
    plot_metrics_comparison
)

logger = logging.getLogger(__name__)

@pipeline_stage(
    # Make parameters optional by removing them from requires list
    provides=["evaluation_results"]
)
class EvaluationStage(PipelineStage):
    """Stage to evaluate all models"""
    
    def __init__(self, results_dir: str = "./results", name: str = "EvaluationStage"):
        super().__init__(name)
        self.results_dir = results_dir
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Evaluate models and calculate metrics"""
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            logger.info(f"Created results directory: {self.results_dir}")
        
        # Get models from context (any might be None if not run)
        single_models = context.get("single_models", {})
        chained_models = context.get("chained_models", {})
        hierarchical_models = context.get("hierarchical_models", {})
        
        # Log what models we found
        if single_models:
            logger.info(f"Found {len(single_models)} single target models for evaluation")
        if chained_models:
            logger.info(f"Found {len(chained_models)} chained models for evaluation")
        if hierarchical_models:
            logger.info(f"Found {len(hierarchical_models)} hierarchical models for evaluation")
        
        # Prepare metrics container
        all_metrics = {
            "single": {},
            "chained": {"accuracy": [], "f1_macro": [], "f1_weighted": [], "groups": {}},
            "hierarchical": {"accuracy": [], "f1_macro": [], "f1_weighted": [], "groups": {}}
        }
        
        # Process chained models
        if chained_models:
            logger.info(f"Evaluating {len(chained_models)} chained models")
            
            # Import necessary metrics calculation libraries
            from sklearn.metrics import (
                accuracy_score, 
                precision_score, 
                recall_score, 
                f1_score, 
                classification_report
            )
            
            for group_name, model_info in chained_models.items():
                model = model_info["model"]
                
                # Get test data and predictions
                if hasattr(model, 'test_features') and hasattr(model, 'test_targets'):
                    X_test = model.test_features
                    y_test = model.test_targets
                    
                    # Make predictions
                    try:
                        predictions = model.predict(X_test)
                        
                        # Initialize metrics for this group
                        group_metrics = {}
                        
                        # Type 2 metrics
                        type2_true = y_test[:, 0] if y_test.ndim > 1 else y_test
                        type2_pred = predictions[:, 0] if predictions.ndim > 1 else predictions
                        
                        # Calculate metrics
                        type2_metrics = {
                            "accuracy": accuracy_score(type2_true, type2_pred),
                            "precision": precision_score(type2_true, type2_pred, average='weighted', zero_division=0),
                            "recall": recall_score(type2_true, type2_pred, average='weighted', zero_division=0),
                            "f1_macro": f1_score(type2_true, type2_pred, average='macro', zero_division=0),
                            "f1_weighted": f1_score(type2_true, type2_pred, average='weighted', zero_division=0),
                            "class_report": classification_report(type2_true, type2_pred, output_dict=True, zero_division=0)
                        }
                        
                        group_metrics["type2"] = type2_metrics
                        all_metrics["chained"]["accuracy"].append(type2_metrics["accuracy"])
                        all_metrics["chained"]["f1_macro"].append(type2_metrics["f1_macro"])
                        all_metrics["chained"]["f1_weighted"].append(type2_metrics["f1_weighted"])
                        
                        # Type 2 + Type 3 metrics if available
                        if y_test.ndim > 1 and y_test.shape[1] >= 2 and predictions.ndim > 1 and predictions.shape[1] >= 2:
                            type2_y3_true = y_test[:, :2]
                            type2_y3_pred = predictions[:, :2]
                            
                            # Calculate combined accuracy
                            type2_y3_accuracy = np.mean(np.all(type2_y3_true == type2_y3_pred, axis=1))
                            
                            group_metrics["type2_y3"] = {
                                "accuracy": type2_y3_accuracy,
                                "f1_macro": 0,  # Placeholder, not easy to calculate for combined targets
                                "f1_weighted": 0
                            }
                        
                        # Type 2 + Type 3 + Type 4 metrics if available
                        if y_test.ndim > 1 and y_test.shape[1] >= 3 and predictions.ndim > 1 and predictions.shape[1] >= 3:
                            type2_y3_y4_true = y_test
                            type2_y3_y4_pred = predictions
                            
                            # Calculate combined accuracy
                            type2_y3_y4_accuracy = np.mean(np.all(type2_y3_y4_true == type2_y3_y4_pred, axis=1))
                            
                            group_metrics["type2_y3_y4"] = {
                                "accuracy": type2_y3_y4_accuracy,
                                "f1_macro": 0,  # Placeholder, not easy to calculate for combined targets
                                "f1_weighted": 0
                            }
                        
                        # Store group metrics
                        all_metrics["chained"]["groups"][group_name] = group_metrics
                        
                    except Exception as e:
                        logger.error(f"Error evaluating chained model for {group_name}: {str(e)}")
                else:
                    logger.warning(f"Model for {group_name} does not have test data. Skipping evaluation.")
        
        # Process hierarchical models
        if hierarchical_models:
            logger.info(f"Evaluating {len(hierarchical_models)} hierarchical models")
            
            # Import necessary metrics calculation libraries if not already imported
            if 'accuracy_score' not in locals():
                from sklearn.metrics import (
                    accuracy_score, 
                    precision_score, 
                    recall_score, 
                    f1_score, 
                    classification_report
                )
            
            for group_name, model_info in hierarchical_models.items():
                base_model = model_info["base_model"]
                all_models = model_info.get("all_models", {})
                
                # Initialize metrics for this group
                group_metrics = {}
                
                # Evaluate base model first
                if hasattr(base_model, 'test_data'):
                    base_test_data = base_model.test_data
                    
                    # Try to extract test features and targets from the model
                    if hasattr(base_model, 'test_features') and hasattr(base_model, 'test_targets'):
                        base_X_test = base_model.test_features
                        base_y_test = base_model.test_targets
                    else:
                        # Extract from test_data
                        base_X_test = base_test_data.X if hasattr(base_test_data, 'X') else None
                        base_y_test = base_test_data.y if hasattr(base_test_data, 'y') else None
                    
                    # Evaluate if we have valid test data
                    if base_X_test is not None and base_y_test is not None and len(base_y_test) > 0:
                        try:
                            # Make predictions
                            base_predictions = base_model.predict(base_X_test)
                            
                            # Calculate metrics
                            base_metrics = {
                                "accuracy": accuracy_score(base_y_test, base_predictions),
                                "precision": precision_score(base_y_test, base_predictions, average='weighted', zero_division=0),
                                "recall": recall_score(base_y_test, base_predictions, average='weighted', zero_division=0),
                                "f1_macro": f1_score(base_y_test, base_predictions, average='macro', zero_division=0),
                                "f1_weighted": f1_score(base_y_test, base_predictions, average='weighted', zero_division=0),
                                "class_report": classification_report(base_y_test, base_predictions, output_dict=True, zero_division=0)
                            }
                            
                            group_metrics["base"] = base_metrics
                            
                            # Add to overall metrics arrays
                            all_metrics["hierarchical"]["accuracy"].append(base_metrics["accuracy"])
                            all_metrics["hierarchical"]["f1_macro"].append(base_metrics["f1_macro"])
                            all_metrics["hierarchical"]["f1_weighted"].append(base_metrics["f1_weighted"])
                            
                        except Exception as e:
                            logger.error(f"Error evaluating base model for {group_name}: {str(e)}")
                    else:
                        logger.warning(f"Base model for {group_name} does not have proper test data. Attempting to use model's stored test data.")
                        
                        # Fallback: try to use any available test data directly from the model
                        if hasattr(base_model, 'X_test') and hasattr(base_model, 'y_test') and \
                           base_model.X_test is not None and base_model.y_test is not None and \
                           len(base_model.y_test) > 0:
                            try:
                                # Make predictions with stored test data
                                fallback_predictions = base_model.predict(base_model.X_test)
                                
                                # Calculate metrics with zero_division=0 to handle sparse data
                                fallback_metrics = {
                                    "accuracy": accuracy_score(base_model.y_test, fallback_predictions),
                                    "precision": precision_score(base_model.y_test, fallback_predictions, average='weighted', zero_division=0),
                                    "recall": recall_score(base_model.y_test, fallback_predictions, average='weighted', zero_division=0),
                                    "f1_macro": f1_score(base_model.y_test, fallback_predictions, average='macro', zero_division=0),
                                    "f1_weighted": f1_score(base_model.y_test, fallback_predictions, average='weighted', zero_division=0),
                                    "class_report": classification_report(base_model.y_test, fallback_predictions, output_dict=True, zero_division=0)
                                }
                                
                                group_metrics["base"] = fallback_metrics
                                
                                # Add to overall metrics arrays
                                all_metrics["hierarchical"]["accuracy"].append(fallback_metrics["accuracy"])
                                all_metrics["hierarchical"]["f1_macro"].append(fallback_metrics["f1_macro"])
                                all_metrics["hierarchical"]["f1_weighted"].append(fallback_metrics["f1_weighted"])
                                
                                logger.info(f"Successfully evaluated base model for {group_name} with fallback test data")
                            except Exception as e:
                                logger.error(f"Error evaluating base model for {group_name} with fallback data: {str(e)}")
                        else:
                            logger.warning(f"Base model for {group_name} does not have any usable test data. Skipping evaluation.")
                else:
                    logger.warning(f"Base model for {group_name} does not have test data. Skipping evaluation.")
                
                # Evaluate child models
                for model_key, model in all_models.items():
                    if hasattr(model, 'test_data'):
                        child_test_data = model.test_data
                        
                        child_X_test = child_test_data.X if hasattr(child_test_data, 'X') else None
                        child_y_test = child_test_data.y if hasattr(child_test_data, 'y') else None
                        
                        if child_X_test is not None and child_y_test is not None and len(child_y_test) > 0:
                            try:
                                # For hierarchical models, we need to be more lenient with sparse data
                                # Calculate metrics even with minimal test samples
                                child_classes = np.unique(child_y_test)
                                
                                # Make predictions
                                child_predictions = model.predict(child_X_test)
                                
                                # Calculate metrics
                                child_metrics = {
                                    "accuracy": accuracy_score(child_y_test, child_predictions),
                                    "precision": precision_score(child_y_test, child_predictions, average='weighted', zero_division=0),
                                    "recall": recall_score(child_y_test, child_predictions, average='weighted', zero_division=0),
                                    "f1_macro": f1_score(child_y_test, child_predictions, average='macro', zero_division=0),
                                    "f1_weighted": f1_score(child_y_test, child_predictions, average='weighted', zero_division=0),
                                    "class_report": classification_report(child_y_test, child_predictions, output_dict=True, zero_division=0)
                                }
                                
                                group_metrics[str(model_key)] = child_metrics
                                
                                # Add to the overall metrics
                                all_metrics["hierarchical"]["accuracy"].append(child_metrics["accuracy"])
                                all_metrics["hierarchical"]["f1_macro"].append(child_metrics["f1_macro"])
                                all_metrics["hierarchical"]["f1_weighted"].append(child_metrics["f1_weighted"])
                                
                            except Exception as e:
                                logger.error(f"Error evaluating child model {model_key} for {group_name}: {str(e)}")
                        else:
                            logger.warning(f"Child model {model_key} for {group_name} has insufficient test data. Attempting to use minimal evaluation.")
                            # Try to use any available test data
                            if hasattr(model, 'test_features') and hasattr(model, 'test_targets') and \
                               model.test_features is not None and model.test_targets is not None and \
                               len(model.test_targets) > 0:
                                try:
                                    # Make predictions with minimal data
                                    minimal_predictions = model.predict(model.test_features)
                                    
                                    # Calculate metrics with zero_division=0 to handle sparse data
                                    minimal_metrics = {
                                        "accuracy": accuracy_score(model.test_targets, minimal_predictions),
                                        "precision": precision_score(model.test_targets, minimal_predictions, average='weighted', zero_division=0),
                                        "recall": recall_score(model.test_targets, minimal_predictions, average='weighted', zero_division=0),
                                        "f1_macro": f1_score(model.test_targets, minimal_predictions, average='macro', zero_division=0),
                                        "f1_weighted": f1_score(model.test_targets, minimal_predictions, average='weighted', zero_division=0),
                                        "class_report": classification_report(model.test_targets, minimal_predictions, output_dict=True, zero_division=0)
                                    }
                                    
                                    group_metrics[str(model_key)] = minimal_metrics
                                    
                                    # Add to the overall metrics
                                    all_metrics["hierarchical"]["accuracy"].append(minimal_metrics["accuracy"])
                                    all_metrics["hierarchical"]["f1_macro"].append(minimal_metrics["f1_macro"])
                                    all_metrics["hierarchical"]["f1_weighted"].append(minimal_metrics["f1_weighted"])
                                    
                                    logger.info(f"Successfully evaluated child model {model_key} with minimal data")
                                except Exception as e:
                                    logger.error(f"Error evaluating child model {model_key} with minimal data: {str(e)}")
                            else:
                                logger.warning(f"Child model {model_key} for {group_name} does not have any test data. Skipping evaluation.")
                
                # Store group metrics
                all_metrics["hierarchical"]["groups"][group_name] = group_metrics
        
        # Process single models (if any)
        if single_models:
            logger.info(f"Evaluating {len(single_models)} single target models")
            for group_name, model_info in single_models.items():
                model = model_info["model"]
                metrics = model_info["metrics"]
                all_metrics["single"][group_name] = metrics
        
        # Get overall metrics 
        overall_metrics = self._calculate_overall_metrics(all_metrics)
        
        # Save metrics to file
        metrics_file = os.path.join(self.results_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(overall_metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Store metrics in context
        context.set("evaluation_results", {
            "all_metrics": all_metrics,
            "overall_metrics": overall_metrics,
            "metrics_file": metrics_file
        })
        
        return context
    
    def _calculate_overall_metrics(self, all_metrics: Dict) -> Dict:
        """Calculate aggregated metrics across all models"""
        overall = {}
        
        # Process each model type
        for model_type, groups in all_metrics.items():
            if not groups:
                continue
                
            overall[model_type] = {
                "accuracy": [],
                "f1_macro": [],
                "f1_weighted": [],
                "groups": {}
            }
            
            # Skip arrays that might be mistakenly identified as groups
            if model_type in ["chained", "hierarchical"] and isinstance(groups, dict) and "groups" in groups:
                # Copy the pre-calculated metrics arrays
                overall[model_type]["accuracy"] = groups["accuracy"]
                overall[model_type]["f1_macro"] = groups["f1_macro"]
                overall[model_type]["f1_weighted"] = groups["f1_weighted"]
                
                # Copy the group metrics directly
                overall[model_type]["groups"] = groups["groups"]
            elif model_type == "single":
                # Process each group
                for group_name, metrics in groups.items():
                    overall[model_type]["groups"][group_name] = {}
                    
                    # Process metrics for this group
                    if isinstance(metrics, dict):
                        # For single models, just copy the metrics
                        overall[model_type]["groups"][group_name] = metrics
                        
                        # Add to aggregate metrics if available
                        if "accuracy" in metrics:
                            overall[model_type]["accuracy"].append(metrics["accuracy"])
                        if "f1_macro" in metrics:
                            overall[model_type]["f1_macro"].append(metrics["f1_macro"])
                        if "f1_weighted" in metrics:
                            overall[model_type]["f1_weighted"].append(metrics["f1_weighted"])
            
            # Calculate averages for each model type if metrics were collected
            if overall[model_type]["accuracy"]:
                overall[model_type]["avg_accuracy"] = sum(overall[model_type]["accuracy"]) / len(overall[model_type]["accuracy"])
                
            if overall[model_type]["f1_macro"]:
                overall[model_type]["avg_f1_macro"] = sum(overall[model_type]["f1_macro"]) / len(overall[model_type]["f1_macro"])
                
            if overall[model_type]["f1_weighted"]:
                overall[model_type]["avg_f1_weighted"] = sum(overall[model_type]["f1_weighted"]) / len(overall[model_type]["f1_weighted"])
        
        return overall

    def _print_detailed_metrics(self, model_type, metrics):
        """
        Print detailed metrics for model evaluation using custom formatting
        
        Args:
            model_type (str): Type of model (chained or hierarchical)
            metrics (dict): Dictionary containing metrics
        """
        logger = logging.getLogger(__name__)
        
        # Create a detailed metrics text file
        detailed_metrics_path = os.path.join(self.results_dir, 'detailed_metrics.txt')
        
        with open(detailed_metrics_path, 'w') as f:
            header_line = "=" * 80
            f.write(f"{header_line}\n")
            f.write(f"{' DETAILED PERFORMANCE METRICS FOR ' + model_type.upper() + ' MODEL ':^80}\n")
            f.write(f"{header_line}\n\n")
            
            # Process each group
            for group, group_metrics in metrics.get(model_type, {}).get('groups', {}).items():
                f.write(f"\n\n{' GROUP: ' + group + ' ':*^80}\n\n")
                
                # For chained model
                if model_type == 'chained':
                    if 'type2' in group_metrics:
                        f.write(f"{' TYPE 2 METRICS ':-^60}\n\n")
                        t2_metrics = group_metrics['type2']
                        
                        # Create a formatted metrics table
                        accuracy = t2_metrics.get('accuracy', 0) * 100
                        f1_macro = t2_metrics.get('f1_macro', 0) * 100
                        f1_weighted = t2_metrics.get('f1_weighted', 0) * 100
                        precision = t2_metrics.get('precision', 0) * 100
                        recall = t2_metrics.get('recall', 0) * 100
                        
                        # Format the key metrics
                        f.write(f"{'Metric':<20} | {'Value':<15} | {'Range':<15} | {'Quality':<15}\n")
                        f.write(f"{'-'*20} | {'-'*15} | {'-'*15} | {'-'*15}\n")
                        f.write(f"{'Accuracy':<20} | {accuracy:15.2f}% | {'0-100%':<15} | {self._quality_rating(accuracy):<15}\n")
                        f.write(f"{'Precision':<20} | {precision:15.2f}% | {'0-100%':<15} | {self._quality_rating(precision):<15}\n")
                        f.write(f"{'Recall':<20} | {recall:15.2f}% | {'0-100%':<15} | {self._quality_rating(recall):<15}\n")
                        f.write(f"{'F1 Score (Macro)':<20} | {f1_macro:15.2f}% | {'0-100%':<15} | {self._quality_rating(f1_macro):<15}\n")
                        f.write(f"{'F1 Score (Weighted)':<20} | {f1_weighted:15.2f}% | {'0-100%':<15} | {self._quality_rating(f1_weighted):<15}\n\n")
                        
                        # Add class-wise metrics if available
                        if 'class_report' in t2_metrics:
                            f.write("\nClass-wise Performance:\n")
                            f.write("-" * 60 + "\n")
                            f.write(f"{t2_metrics['class_report']}\n")
                    
                    # Add other levels metrics if available
                    if 'type2_type3' in group_metrics:
                        f.write(f"\n{' TYPE 2 + TYPE 3 METRICS ':-^60}\n\n")
                        t3_metrics = group_metrics['type2_type3']
                        
                        accuracy = t3_metrics.get('accuracy', 0) * 100
                        f1_macro = t3_metrics.get('f1_macro', 0) * 100
                        f1_weighted = t3_metrics.get('f1_weighted', 0) * 100
                        
                        # Format the key metrics
                        f.write(f"{'Metric':<20} | {'Value':<15} | {'Quality':<15}\n")
                        f.write(f"{'-'*20} | {'-'*15} | {'-'*15}\n")
                        f.write(f"{'Accuracy':<20} | {accuracy:15.2f}% | {self._quality_rating(accuracy):<15}\n")
                        f.write(f"{'F1 Score (Macro)':<20} | {f1_macro:15.2f}% | {self._quality_rating(f1_macro):<15}\n")
                        f.write(f"{'F1 Score (Weighted)':<20} | {f1_weighted:15.2f}% | {self._quality_rating(f1_weighted):<15}\n\n")
                    
                    if 'type2_type3_type4' in group_metrics:
                        f.write(f"\n{' TYPE 2 + TYPE 3 + TYPE 4 METRICS ':-^60}\n\n")
                        t4_metrics = group_metrics['type2_type3_type4']
                        
                        accuracy = t4_metrics.get('accuracy', 0) * 100
                        f1_macro = t4_metrics.get('f1_macro', 0) * 100
                        f1_weighted = t4_metrics.get('f1_weighted', 0) * 100
                        
                        # Format the key metrics
                        f.write(f"{'Metric':<20} | {'Value':<15} | {'Quality':<15}\n")
                        f.write(f"{'-'*20} | {'-'*15} | {'-'*15}\n")
                        f.write(f"{'Accuracy':<20} | {accuracy:15.2f}% | {self._quality_rating(accuracy):<15}\n")
                        f.write(f"{'F1 Score (Macro)':<20} | {f1_macro:15.2f}% | {self._quality_rating(f1_macro):<15}\n")
                        f.write(f"{'F1 Score (Weighted)':<20} | {f1_weighted:15.2f}% | {self._quality_rating(f1_weighted):<15}\n\n")
                
                # For hierarchical model
                else:
                    # Base model (Type 2)
                    if 'base' in group_metrics:
                        f.write(f"{' TYPE 2 BASE MODEL METRICS ':-^60}\n\n")
                        base_metrics = group_metrics['base']
                        
                        # Create a formatted metrics table
                        accuracy = base_metrics.get('accuracy', 0) * 100
                        f1_macro = base_metrics.get('f1_macro', 0) * 100
                        f1_weighted = base_metrics.get('f1_weighted', 0) * 100
                        precision = base_metrics.get('precision', 0) * 100
                        recall = base_metrics.get('recall', 0) * 100
                        
                        # Format the key metrics
                        f.write(f"{'Metric':<20} | {'Value':<15} | {'Range':<15} | {'Quality':<15}\n")
                        f.write(f"{'-'*20} | {'-'*15} | {'-'*15} | {'-'*15}\n")
                        f.write(f"{'Accuracy':<20} | {accuracy:15.2f}% | {'0-100%':<15} | {self._quality_rating(accuracy):<15}\n")
                        f.write(f"{'Precision':<20} | {precision:15.2f}% | {'0-100%':<15} | {self._quality_rating(precision):<15}\n")
                        f.write(f"{'Recall':<20} | {recall:15.2f}% | {'0-100%':<15} | {self._quality_rating(recall):<15}\n")
                        f.write(f"{'F1 Score (Macro)':<20} | {f1_macro:15.2f}% | {'0-100%':<15} | {self._quality_rating(f1_macro):<15}\n")
                        f.write(f"{'F1 Score (Weighted)':<20} | {f1_weighted:15.2f}% | {'0-100%':<15} | {self._quality_rating(f1_weighted):<15}\n\n")
                        
                        # Add class-wise metrics if available
                        if 'class_report' in base_metrics:
                            f.write("\nClass-wise Performance:\n")
                            f.write("-" * 60 + "\n")
                            f.write(f"{base_metrics['class_report']}\n")
                    
                    # Child models for each Type 2 class
                    for model_key, model_metrics in group_metrics.items():
                        if model_key != 'base' and isinstance(model_metrics, dict):
                            f.write(f"\n{' TYPE 3 MODEL FOR ' + model_key + ' METRICS ':-^60}\n\n")
                            
                            # Create a formatted metrics table
                            accuracy = model_metrics.get('accuracy', 0) * 100
                            f1_macro = model_metrics.get('f1_macro', 0) * 100
                            f1_weighted = model_metrics.get('f1_weighted', 0) * 100
                            
                            # Format the key metrics
                            f.write(f"{'Metric':<20} | {'Value':<15} | {'Quality':<15}\n")
                            f.write(f"{'-'*20} | {'-'*15} | {'-'*15}\n")
                            f.write(f"{'Accuracy':<20} | {accuracy:15.2f}% | {self._quality_rating(accuracy):<15}\n")
                            f.write(f"{'F1 Score (Macro)':<20} | {f1_macro:15.2f}% | {self._quality_rating(f1_macro):<15}\n")
                            f.write(f"{'F1 Score (Weighted)':<20} | {f1_weighted:15.2f}% | {self._quality_rating(f1_weighted):<15}\n\n")
                            
                            # Type 4 models (if any)
                            for sub_key, sub_metrics in model_metrics.items():
                                if isinstance(sub_metrics, dict) and sub_key != 'class_report':
                                    f.write(f"\n{' TYPE 4 MODEL FOR ' + model_key + ', ' + sub_key + ' METRICS ':-^60}\n\n")
                                    
                                    # Create a formatted metrics table
                                    accuracy = sub_metrics.get('accuracy', 0) * 100
                                    f1_macro = sub_metrics.get('f1_macro', 0) * 100
                                    f1_weighted = sub_metrics.get('f1_weighted', 0) * 100
                                    
                                    # Format the key metrics
                                    f.write(f"{'Metric':<20} | {'Value':<15} | {'Quality':<15}\n")
                                    f.write(f"{'-'*20} | {'-'*15} | {'-'*15}\n")
                                    f.write(f"{'Accuracy':<20} | {accuracy:15.2f}% | {self._quality_rating(accuracy):<15}\n")
                                    f.write(f"{'F1 Score (Macro)':<20} | {f1_macro:15.2f}% | {self._quality_rating(f1_macro):<15}\n")
                                    f.write(f"{'F1 Score (Weighted)':<20} | {f1_weighted:15.2f}% | {self._quality_rating(f1_weighted):<15}\n\n")
        
        # Overall performance summary
        f.write(f"\n\n{header_line}\n")
        f.write(f"{' OVERALL PERFORMANCE SUMMARY ':^80}\n")
        f.write(f"{header_line}\n\n")
        
        # Calculate average metrics across groups
        avg_accuracy = np.mean([group_metrics.get('type2', {}).get('accuracy', 0) * 100
                               for group_metrics in metrics.get(model_type, {}).get('groups', {}).values()
                               if 'type2' in group_metrics])
        
        avg_f1_macro = np.mean([group_metrics.get('type2', {}).get('f1_macro', 0) * 100
                               for group_metrics in metrics.get(model_type, {}).get('groups', {}).values()
                               if 'type2' in group_metrics])
        
        avg_f1_weighted = np.mean([group_metrics.get('type2', {}).get('f1_weighted', 0) * 100
                                  for group_metrics in metrics.get(model_type, {}).get('groups', {}).values()
                                  if 'type2' in group_metrics])
        
        # Format the summary metrics
        f.write(f"{'Metric':<30} | {'Value':<15} | {'Quality':<15}\n")
        f.write(f"{'-'*30} | {'-'*15} | {'-'*15}\n")
        f.write(f"{'Average Type 2 Accuracy':<30} | {avg_accuracy:15.2f}% | {self._quality_rating(avg_accuracy):<15}\n")
        f.write(f"{'Average Type 2 F1 (Macro)':<30} | {avg_f1_macro:15.2f}% | {self._quality_rating(avg_f1_macro):<15}\n")
        f.write(f"{'Average Type 2 F1 (Weighted)':<30} | {avg_f1_weighted:15.2f}% | {self._quality_rating(avg_f1_weighted):<15}\n")
        
        # Add model complexity information
        model_count = len(metrics.get(model_type, {}).get('groups', {}))
        f.write(f"\n\n{' MODEL COMPLEXITY ':-^60}\n\n")
        f.write(f"Total number of {model_type} models: {model_count}\n")
        
        # Model-specific complexity metrics
        if model_type == 'hierarchical':
            # Count child models
            child_model_count = 0
            for group_name, group_data in metrics.get(model_type, {}).get('groups', {}).items():
                for key, value in group_data.items():
                    if key != 'base' and isinstance(value, dict):
                        child_model_count += 1
            
            f.write(f"Total number of child models: {child_model_count}\n")
            f.write(f"Total model count (base + child): {model_count + child_model_count}\n")
        
        # Add model architecture recommendation
        f.write(f"\n\n{' RECOMMENDATION ':-^60}\n\n")
        if avg_accuracy >= 80:
            f.write(f"The {model_type} model shows EXCELLENT performance with an average accuracy of {avg_accuracy:.2f}%.\n")
            f.write("This model architecture is recommended for production use.\n")
        elif avg_accuracy >= 70:
            f.write(f"The {model_type} model shows GOOD performance with an average accuracy of {avg_accuracy:.2f}%.\n")
            f.write("This model architecture is suitable for most applications, but may benefit from further fine-tuning.\n")
        elif avg_accuracy >= 60:
            f.write(f"The {model_type} model shows MODERATE performance with an average accuracy of {avg_accuracy:.2f}%.\n")
            f.write("Consider enhancing the model with more training data or feature engineering.\n")
        else:
            f.write(f"The {model_type} model shows BELOW AVERAGE performance with an accuracy of {avg_accuracy:.2f}%.\n")
            f.write("Significant improvements are needed before production deployment.\n")
        
        logger.info(f"Generated detailed metrics report at {detailed_metrics_path}")
        return detailed_metrics_path

    def _quality_rating(self, value):
        """
        Return a quality rating based on the metric value
        
        Args:
            value (float): Metric value as a percentage (0-100)
            
        Returns:
            str: Quality rating
        """
        if value >= 90:
            return "Excellent"
        elif value >= 80:
            return "Very Good"
        elif value >= 70:
            return "Good"
        elif value >= 60:
            return "Moderate"
        elif value >= 50:
            return "Fair"
        else:
            return "Poor"

    def _evaluate_models(self, model_type):
        """
        Evaluate all models of the given type.
        
        Args:
            model_type (str): Type of model to evaluate ('chained' or 'hierarchical')
        """
        logger = logging.getLogger(__name__)
        
        if model_type == 'chained':
            models = self._get_chained_models()
            logger.info(f"Found {len(models)} chained models for evaluation")
        else:
            models = self._get_hierarchical_models()
            logger.info(f"Found {len(models)} hierarchical models for evaluation")

        if not models:
            logger.warning(f"No {model_type} models found to evaluate")
            return {}

        # Perform evaluation based on model type
        metrics = {model_type: {"accuracy": [], "f1_macro": [], "f1_weighted": [], "groups": {}}}
        
        logger.info(f"Evaluating {len(models)} {model_type} models")
        
        # Import necessary libraries for evaluation
        from sklearn.metrics import (
            accuracy_score, 
            precision_score, 
            recall_score, 
            f1_score, 
            classification_report
        )
        
        for model_name, model in models:
            group_name = model_name.split('_', 1)[1]
            
            if model_type == 'chained':
                # For chained models, evaluate each level
                if not hasattr(model, 'test_features') or not hasattr(model, 'test_targets'):
                    logger.warning(f"Model {model_name} does not have test data. Skipping evaluation.")
                    continue
                
                X_test = model.test_features
                y_test = model.test_targets
                
                # Make predictions
                try:
                    predictions = model.predict(X_test)
                    
                    # Initialize metrics for this group
                    group_metrics = {}
                    
                    # Type 2 metrics
                    type2_true = y_test[:, 0] if y_test.ndim > 1 else y_test
                    type2_pred = predictions[:, 0] if predictions.ndim > 1 else predictions
                    
                    # Calculate metrics
                    type2_metrics = {
                        "accuracy": accuracy_score(type2_true, type2_pred),
                        "precision": precision_score(type2_true, type2_pred, average='weighted', zero_division=0),
                        "recall": recall_score(type2_true, type2_pred, average='weighted', zero_division=0),
                        "f1_macro": f1_score(type2_true, type2_pred, average='macro', zero_division=0),
                        "f1_weighted": f1_score(type2_true, type2_pred, average='weighted', zero_division=0),
                        "class_report": classification_report(type2_true, type2_pred, output_dict=True, zero_division=0)
                    }
                    
                    group_metrics["type2"] = type2_metrics
                    metrics[model_type]["accuracy"].append(type2_metrics["accuracy"])
                    metrics[model_type]["f1_macro"].append(type2_metrics["f1_macro"])
                    metrics[model_type]["f1_weighted"].append(type2_metrics["f1_weighted"])
                    
                    # Type 2 + Type 3 metrics if available
                    if y_test.ndim > 1 and y_test.shape[1] >= 2 and predictions.ndim > 1 and predictions.shape[1] >= 2:
                        type2_y3_true = y_test[:, :2]
                        type2_y3_pred = predictions[:, :2]
                        
                        # Calculate combined accuracy
                        type2_y3_accuracy = np.mean(np.all(type2_y3_true == type2_y3_pred, axis=1))
                        
                        group_metrics["type2_y3"] = {
                            "accuracy": type2_y3_accuracy,
                            "f1_macro": 0,  # Placeholder, not easy to calculate for combined targets
                            "f1_weighted": 0
                        }
                    
                    # Type 2 + Type 3 + Type 4 metrics if available
                    if y_test.ndim > 1 and y_test.shape[1] >= 3 and predictions.ndim > 1 and predictions.shape[1] >= 3:
                        type2_y3_y4_true = y_test
                        type2_y3_y4_pred = predictions
                        
                        # Calculate combined accuracy
                        type2_y3_y4_accuracy = np.mean(np.all(type2_y3_y4_true == type2_y3_y4_pred, axis=1))
                        
                        group_metrics["type2_y3_y4"] = {
                            "accuracy": type2_y3_y4_accuracy,
                            "f1_macro": 0,  # Placeholder, not easy to calculate for combined targets
                            "f1_weighted": 0
                        }
                    
                    # Store group metrics
                    metrics[model_type]["groups"][group_name] = group_metrics
                    
                except Exception as e:
                    logger.error(f"Error evaluating chained model {model_name}: {str(e)}")
                    
            else:  # Hierarchical model
                # For hierarchical models, evaluate each level separately
                if not hasattr(model, 'base_model') or not hasattr(model, 'test_data'):
                    logger.warning(f"Model {model_name} does not have proper structure or test data. Skipping evaluation.")
                    continue
                
                # Initialize metrics for this group
                group_metrics = {}
                
                try:
                    # Evaluate base model (Type 2)
                    base_model = model.base_model
                    test_data = model.test_data
                    
                    X_test = test_data.X
                    y_test = test_data.y
                    
                    # Make predictions
                    base_predictions = base_model.predict(X_test)
                    
                    # Calculate metrics
                    base_metrics = {
                        "accuracy": accuracy_score(y_test, base_predictions),
                        "precision": precision_score(y_test, base_predictions, average='weighted', zero_division=0),
                        "recall": recall_score(y_test, base_predictions, average='weighted', zero_division=0),
                        "f1_macro": f1_score(y_test, base_predictions, average='macro', zero_division=0),
                        "f1_weighted": f1_score(y_test, base_predictions, average='weighted', zero_division=0),
                        "class_report": classification_report(y_test, base_predictions, output_dict=True, zero_division=0)
                    }
                    
                    group_metrics["base"] = base_metrics
                    metrics[model_type]["accuracy"].append(base_metrics["accuracy"])
                    metrics[model_type]["f1_macro"].append(base_metrics["f1_macro"])
                    metrics[model_type]["f1_weighted"].append(base_metrics["f1_weighted"])
                    
                    # Evaluate child models (Type 3)
                    if hasattr(model, 'child_models'):
                        for type2_class, child_model in model.child_models.items():
                            if not hasattr(child_model, 'test_data'):
                                continue
                                
                            child_test_data = child_model.test_data
                            child_X_test = child_test_data.X
                            child_y_test = child_test_data.y
                            
                            # Make predictions
                            child_predictions = child_model.predict(child_X_test)
                            
                            # Calculate metrics
                            if child_X_test is not None and child_y_test is not None and len(child_y_test) > 0:
                                child_metrics = {
                                    "accuracy": accuracy_score(child_y_test, child_predictions),
                                    "precision": precision_score(child_y_test, child_predictions, average='weighted', zero_division=0),
                                    "recall": recall_score(child_y_test, child_predictions, average='weighted', zero_division=0),
                                    "f1_macro": f1_score(child_y_test, child_predictions, average='macro', zero_division=0),
                                    "f1_weighted": f1_score(child_y_test, child_predictions, average='weighted', zero_division=0),
                                    "class_report": classification_report(child_y_test, child_predictions, output_dict=True, zero_division=0)
                                }
                            else:
                                child_metrics = {
                                    "accuracy": 0,
                                    "precision": 0,
                                    "recall": 0,
                                    "f1_macro": 0,
                                    "f1_weighted": 0
                                }
                            
                            group_metrics[str(type2_class)] = child_metrics
                    
                    # Store group metrics
                    metrics[model_type]["groups"][group_name] = group_metrics
                    
                except Exception as e:
                    logger.error(f"Error evaluating hierarchical model {model_name}: {str(e)}")
        
        # Save metrics to file
        metrics_path = os.path.join(self.results_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Generate detailed metrics report
        self._print_detailed_metrics(model_type, metrics)
        
        return metrics

    def _get_chained_models(self):
        """Get all chained models from saved models"""
        import os
        import glob
        import pickle
        
        models = []
        saved_models_dir = "./saved_models"
        
        if not os.path.exists(saved_models_dir):
            return models
            
        # Find all chained model files
        model_files = glob.glob(os.path.join(saved_models_dir, "ChainedModel_*.pkl"))
        
        for model_file in model_files:
            try:
                # Extract model name from filename
                model_name = os.path.basename(model_file).replace(".pkl", "")
                
                # Load the model
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    
                # Add to list of models
                models.append((model_name, model))
                
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {str(e)}")
                
        return models
        
    def _get_hierarchical_models(self):
        """Get all hierarchical models from saved models"""
        import os
        import glob
        import pickle
        
        models = []
        saved_models_dir = "./saved_models"
        
        if not os.path.exists(saved_models_dir):
            return models
            
        # Find all hierarchical model files
        model_files = glob.glob(os.path.join(saved_models_dir, "HierarchicalModel_*.pkl"))
        
        for model_file in model_files:
            try:
                # Extract model name from filename
                model_name = os.path.basename(model_file).replace(".pkl", "")
                
                # Load the model
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    
                # Add to list of models
                models.append((model_name, model))
                
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {str(e)}")
                
        return models


@pipeline_stage(
    requires=["evaluation_results"],
    provides=["report_file"]
)
class ReportingStage(PipelineStage):
    """Stage to generate a detailed report of results"""
    
    def __init__(self, results_dir: str = "./results", name: str = "ReportingStage"):
        super().__init__(name)
        self.results_dir = results_dir
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Generate a detailed comparison report of the two design decisions"""
        evaluation_results = context.get("evaluation_results", {})
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            logger.info(f"Created results directory: {self.results_dir}")
        
        # Create a report file
        report_file = os.path.join(self.results_dir, "comparison_report.md")
        
        with open(report_file, 'w') as f:
            # Write report header
            f.write("# Email Classification Model Comparison Report\n\n")
            
            # Write pipeline metrics
            f.write("## Pipeline Performance Metrics\n\n")
            pipeline_metrics = [
                ("Total Execution Time", f"{context.get_elapsed_time():.2f} seconds"),
            ]
            
            for name, value in context.metrics.items():
                if name != "total_execution_time":  # Skip this as we use get_elapsed_time
                    # Format numeric values with 2 decimal places
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                    else:
                        formatted_value = str(value)
                    pipeline_metrics.append((name, formatted_value))
            
            # Create a table for pipeline metrics
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for name, value in pipeline_metrics:
                f.write(f"| {name} | {value} |\n")
            f.write("\n")
            
            # Write model comparison section
            f.write("## Model Performance Comparison\n\n")
            
            overall_metrics = evaluation_results.get("overall_metrics", {})
            
            # Compare chained vs hierarchical if both exist
            if "chained" in overall_metrics and "hierarchical" in overall_metrics:
                f.write("### Design Decision 1 vs Design Decision 2\n\n")
                
                # Create comparison table
                f.write("| Metric | Chained Multi-outputs | Hierarchical Modeling |\n")
                f.write("|--------|----------------------|----------------------|\n")
                
                # Add average metrics
                metrics_to_compare = [
                    ("Average Accuracy", "avg_accuracy"),
                    ("Average F1 (Macro)", "avg_f1_macro"),
                    ("Average F1 (Weighted)", "avg_f1_weighted")
                ]
                
                for label, key in metrics_to_compare:
                    chained_value = overall_metrics["chained"].get(key, "N/A")
                    hierarchical_value = overall_metrics["hierarchical"].get(key, "N/A")
                    
                    # Format as percentage if numeric
                    if isinstance(chained_value, (int, float)):
                        chained_value = f"{chained_value*100:.2f}%"
                    if isinstance(hierarchical_value, (int, float)):
                        hierarchical_value = f"{hierarchical_value*100:.2f}%"
                        
                    f.write(f"| {label} | {chained_value} | {hierarchical_value} |\n")
                
                # Add training efficiency comparison if available
                if "chained" in context.metrics and "hierarchical" in context.metrics:
                    chained_time = context.metrics.get("chained_training_time", "N/A")
                    hierarchical_time = context.metrics.get("hierarchical_training_time", "N/A")
                    
                    if isinstance(chained_time, (int, float)) and isinstance(hierarchical_time, (int, float)):
                        f.write(f"| Training Time | {chained_time:.2f}s | {hierarchical_time:.2f}s |\n")
                
                f.write("\n")
                
            # Write detailed metrics for each model type
            for model_type in ["single", "chained", "hierarchical"]:
                if model_type in overall_metrics:
                    f.write(f"### {model_type.capitalize()} Model Results\n\n")
                    
                    # Write group-level results
                    groups = overall_metrics[model_type].get("groups", {})
                    for group_name, metrics in groups.items():
                        f.write(f"#### Group: {group_name}\n\n")
                        
                        if model_type == "chained":
                            # For chained models, show each level's performance
                            f.write("| Level | Accuracy | F1 (Macro) | F1 (Weighted) |\n")
                            f.write("|-------|----------|------------|---------------|\n")
                            
                            levels = ["type2", "type2_y3", "type2_y3_y4"]
                            level_names = ["Type 2", "Type 2 + Type 3", "Type 2 + Type 3 + Type 4"]
                            
                            for level, name in zip(levels, level_names):
                                if level in metrics:
                                    level_metrics = metrics[level]
                                    accuracy = f"{level_metrics.get('accuracy', 0)*100:.2f}%"
                                    f1_macro = f"{level_metrics.get('f1_macro', 0)*100:.2f}%"
                                    f1_weighted = f"{level_metrics.get('f1_weighted', 0)*100:.2f}%"
                                    
                                    f.write(f"| {name} | {accuracy} | {f1_macro} | {f1_weighted} |\n")
                        
                        elif model_type == "hierarchical":
                            # For hierarchical models, show each model's performance
                            f.write("| Model | Accuracy | F1 (Macro) | F1 (Weighted) |\n")
                            f.write("|-------|----------|------------|---------------|\n")
                            
                            for key, model_metrics in metrics.items():
                                model_name = key if key != "base" else "Type 2 (Base)"
                                
                                # Format metrics
                                accuracy = model_metrics.get("accuracy", 0)
                                f1_macro = model_metrics.get("f1_macro", 0)
                                f1_weighted = model_metrics.get("f1_weighted", 0)
                                
                                # Convert to percentages
                                if isinstance(accuracy, (int, float)):
                                    accuracy = f"{accuracy*100:.2f}%"
                                if isinstance(f1_macro, (int, float)):
                                    f1_macro = f"{f1_macro*100:.2f}%"
                                if isinstance(f1_weighted, (int, float)):
                                    f1_weighted = f"{f1_weighted*100:.2f}%"
                                    
                                f.write(f"| {model_name} | {accuracy} | {f1_macro} | {f1_weighted} |\n")
                        
                        else:
                            # For single models, just show the metrics
                            f.write("| Metric | Value |\n")
                            f.write("|--------|-------|\n")
                            
                            # Format metrics
                            for metric, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    value = f"{value*100:.2f}%" if metric in ["accuracy", "f1_macro", "f1_weighted"] else f"{value:.4f}"
                                f.write(f"| {metric} | {value} |\n")
                        
                        f.write("\n")
            
            # Write conclusion section
            f.write("## Conclusion\n\n")
            f.write("This report compares the performance of two architectural approaches for multi-label email classification:\n\n")
            f.write("1. **Design Decision 1: Chained Multi-outputs** - Using a single model instance for each combination of labels\n")
            f.write("2. **Design Decision 2: Hierarchical Modeling** - Using multiple model instances in a tree structure\n\n")
            
            # Add a brief comparison based on metrics
            if "chained" in overall_metrics and "hierarchical" in overall_metrics:
                chained_acc = overall_metrics["chained"].get("avg_accuracy", 0)
                hierarchical_acc = overall_metrics["hierarchical"].get("avg_accuracy", 0)
                
                if chained_acc > hierarchical_acc:
                    f.write("Based on the average accuracy, **Design Decision 1: Chained Multi-outputs** performs better. ")
                    f.write(f"It achieves an average accuracy of {chained_acc*100:.2f}% compared to {hierarchical_acc*100:.2f}% ")
                    f.write("for the hierarchical approach.\n\n")
                elif hierarchical_acc > chained_acc:
                    f.write("Based on the average accuracy, **Design Decision 2: Hierarchical Modeling** performs better. ")
                    f.write(f"It achieves an average accuracy of {hierarchical_acc*100:.2f}% compared to {chained_acc*100:.2f}% ")
                    f.write("for the chained approach.\n\n")
                else:
                    f.write("Both design decisions show similar performance in terms of average accuracy.\n\n")
            
            # Add trade-offs
            f.write("### Trade-offs\n\n")
            f.write("**Design Decision 1: Chained Multi-outputs**\n")
            f.write("- **Pros**: Simpler architecture, fewer models to train, consistent interface\n")
            f.write("- **Cons**: May not capture hierarchical dependencies as well, less interpretability\n\n")
            
            f.write("**Design Decision 2: Hierarchical Modeling**\n")
            f.write("- **Pros**: Better captures hierarchical relationships, more interpretable results per class\n")
            f.write("- **Cons**: More complex architecture, more models to train and maintain\n\n")
            
            # Final recommendation
            f.write("### Recommendation\n\n")
            if "chained" in overall_metrics and "hierarchical" in overall_metrics:
                if chained_acc > hierarchical_acc:
                    f.write("For this specific email classification task, the **Chained Multi-outputs** approach is recommended ")
                    f.write("due to its better performance and simpler architecture.\n")
                elif hierarchical_acc > chained_acc:
                    f.write("For this specific email classification task, the **Hierarchical Modeling** approach is recommended ")
                    f.write("due to its better performance and ability to capture hierarchical relationships.\n")
                else:
                    f.write("Both approaches show similar performance. The choice between them should be based on other factors ")
                    f.write("such as interpretability needs, maintenance requirements, and specific business contexts.\n")
        
        logger.info(f"Generated comparison report at {report_file}")
        context.set("report_file", report_file)
        
        return context


@pipeline_stage(
    requires=["evaluation_results"],
    provides=["visualization_files"]
)
class VisualizationStage(PipelineStage):
    """Stage to create visualizations of results"""
    
    def __init__(self, results_dir: str = "./results", name: str = "VisualizationStage"):
        super().__init__(name)
        self.results_dir = results_dir
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Generate visualizations of the results"""
        evaluation_results = context.get("evaluation_results", {})
        overall_metrics = evaluation_results.get("overall_metrics", {})
        
        # Create visualization directory if it doesn't exist
        viz_dir = os.path.join(self.results_dir, "visualizations")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
            logger.info(f"Created visualization directory: {viz_dir}")
        
        visualization_files = []
        
        # Generate performance comparison chart
        if "chained" in overall_metrics and "hierarchical" in overall_metrics:
            comparison_file = os.path.join(viz_dir, "decision_comparison.png")
            self._create_comparison_chart(overall_metrics, comparison_file)
            visualization_files.append(comparison_file)
            
        # Generate chained model level comparison chart if available
        if "chained" in overall_metrics and "groups" in overall_metrics["chained"]:
            chained_file = os.path.join(viz_dir, "chained_levels.png")
            self._create_chained_levels_chart(overall_metrics["chained"], chained_file)
            visualization_files.append(chained_file)
            
        # Generate hierarchical model structure visualization if available
        if "hierarchical" in overall_metrics and "groups" in overall_metrics["hierarchical"]:
            hierarchical_file = os.path.join(viz_dir, "hierarchical_structure.png")
            self._create_hierarchical_structure(overall_metrics["hierarchical"], hierarchical_file)
            visualization_files.append(hierarchical_file)
        
        # Store visualization files in context
        context.set("visualization_files", visualization_files)
        logger.info(f"Generated {len(visualization_files)} visualization files")
        
        return context
    
    def _create_comparison_chart(self, overall_metrics: Dict, output_file: str) -> None:
        """Create a bar chart comparing chained and hierarchical approaches"""
        # Get metrics for comparison
        metrics = ["avg_accuracy", "avg_f1_macro", "avg_f1_weighted"]
        metric_labels = ["Average Accuracy", "Average F1 (Macro)", "Average F1 (Weighted)"]
        
        chained_values = [overall_metrics["chained"].get(m, 0) for m in metrics]
        hierarchical_values = [overall_metrics["hierarchical"].get(m, 0) for m in metrics]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, chained_values, width, label='Chained Multi-outputs')
        plt.bar(x + width/2, hierarchical_values, width, label='Hierarchical Modeling')
        
        plt.ylabel('Score')
        plt.title('Performance Comparison: Chained vs Hierarchical')
        plt.xticks(x, metric_labels)
        plt.legend()
        
        # Add values on top of bars
        for i, v in enumerate(chained_values):
            plt.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')
            
        for i, v in enumerate(hierarchical_values):
            plt.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')
        
        # Save chart
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Created comparison chart: {output_file}")
    
    def _create_chained_levels_chart(self, chained_metrics: Dict, output_file: str) -> None:
        """Create a chart showing performance at different chained levels"""
        groups = chained_metrics.get("groups", {})
        
        # Collect metrics for each group and level
        group_names = []
        type2_accuracy = []
        type2_y3_accuracy = []
        type2_y3_y4_accuracy = []
        
        for group_name, metrics in groups.items():
            group_names.append(group_name)
            
            # Get accuracy for each level
            if "type2" in metrics:
                type2_accuracy.append(metrics["type2"].get("accuracy", 0))
            else:
                type2_accuracy.append(0)
                
            if "type2_y3" in metrics:
                type2_y3_accuracy.append(metrics["type2_y3"].get("accuracy", 0))
            else:
                type2_y3_accuracy.append(0)
                
            if "type2_y3_y4" in metrics:
                type2_y3_y4_accuracy.append(metrics["type2_y3_y4"].get("accuracy", 0))
            else:
                type2_y3_y4_accuracy.append(0)
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        x = np.arange(len(group_names))
        width = 0.25
        
        plt.bar(x - width, type2_accuracy, width, label='Type 2')
        plt.bar(x, type2_y3_accuracy, width, label='Type 2 + Type 3')
        plt.bar(x + width, type2_y3_y4_accuracy, width, label='Type 2 + Type 3 + Type 4')
        
        plt.ylabel('Accuracy')
        plt.title('Chained Model Performance at Different Levels')
        plt.xticks(x, group_names, rotation=45, ha='right')
        plt.legend()
        
        # Add grid for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Created chained levels chart: {output_file}")
    
    def _create_hierarchical_structure(self, hierarchical_metrics: Dict, output_file: str) -> None:
        """Create a visualization of the hierarchical model structure"""
        try:
            import networkx as nx
            
            groups = hierarchical_metrics.get("groups", {})
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes and edges for each group
            for group_name, metrics in groups.items():
                # Add group node
                group_node = f"Group: {group_name}"
                G.add_node(group_node, level=0)
                
                # Process all model keys
                for model_key, model_metrics in metrics.items():
                    if model_key == "base":
                        # Add base model node (Type 2)
                        base_node = f"{group_name}: Type 2"
                        G.add_node(base_node, level=1)
                        G.add_edge(group_node, base_node)
                        
                        # Store accuracy for node size
                        accuracy = model_metrics.get("accuracy", 0)
                        G.nodes[base_node]["accuracy"] = accuracy
                    elif "=" in model_key:
                        # Parse the model key to understand the hierarchy
                        parts = model_key.split("_")
                        
                        if len(parts) == 2:
                            # This is a Type 3 model
                            type2_class = parts[0].split("=")[1]
                            type3_node = f"{group_name}: {type2_class} → Type 3"
                            
                            # Add node and edges
                            G.add_node(type3_node, level=2)
                            G.add_edge(f"{group_name}: Type 2", type3_node)
                            
                            # Store accuracy for node size
                            accuracy = model_metrics.get("accuracy", 0)
                            G.nodes[type3_node]["accuracy"] = accuracy
                            
                        elif len(parts) == 3:
                            # This is a Type 4 model
                            type2_class = parts[0].split("=")[1]
                            type3_class = parts[1].split("=")[1]
                            type4_node = f"{group_name}: {type2_class} → {type3_class} → Type 4"
                            type3_node = f"{group_name}: {type2_class} → Type 3"
                            
                            # Add node and edges
                            G.add_node(type4_node, level=3)
                            G.add_edge(type3_node, type4_node)
                            
                            # Store accuracy for node size
                            accuracy = model_metrics.get("accuracy", 0)
                            G.nodes[type4_node]["accuracy"] = accuracy
            
            # Create the plot
            plt.figure(figsize=(14, 10))
            
            # Use hierarchical layout
            pos = nx.multipartite_layout(G, subset_key="level")
            
            # Node sizes based on accuracy
            node_sizes = []
            for node in G.nodes():
                accuracy = G.nodes[node].get("accuracy", 0)
                # Scale node size (min size 300, max size 1500)
                node_sizes.append(300 + accuracy * 1200)
            
            # Draw the graph
            nx.draw(G, pos, with_labels=True, node_size=node_sizes, 
                    node_color='skyblue', font_size=8, arrows=True, 
                    arrowsize=15, edge_color='gray')
            
            plt.title("Hierarchical Model Structure")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Created hierarchical structure visualization: {output_file}")
        except ImportError:
            logger.warning("Could not create hierarchical structure visualization: networkx module not found")
            # Create a simple placeholder image with a message
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Hierarchical Structure Visualization\n(networkx module not installed)", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Created placeholder for hierarchical structure visualization: {output_file}")
        except Exception as e:
            logger.error(f"Error creating hierarchical structure visualization: {str(e)}")
            # Create a simple error message image
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error creating visualization:\n{str(e)}", 
                    horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
            plt.axis('off')
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Created error message for hierarchical structure visualization: {output_file}") 