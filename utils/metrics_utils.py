"""
Metrics utilities for email classification
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from typing import Dict, List, Any, Union, Optional, Tuple


def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute classification metrics for predicted labels.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method for multiclass metrics ('micro', 'macro', 'weighted')

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def parse_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Parse the sklearn classification report into a structured dictionary.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Class names (if None, unique values from y_true will be used)

    Returns:
        Nested dictionary with metrics for each class
    """
    # Get classification report as dictionary
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Remove unnecessary string version
    if 'weighted avg' in report:
        report['weighted_avg'] = report.pop('weighted avg')
    if 'macro avg' in report:
        report['macro_avg'] = report.pop('macro avg')
    
    return report


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Calculate both overall and per-class metrics for multiclass classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names (if None, unique labels will be used)

    Returns:
        Tuple containing (overall_metrics, per_class_metrics)
    """
    # Get overall metrics
    overall_metrics = get_classification_metrics(y_true, y_pred)
    
    # Get per-class metrics
    per_class_metrics = parse_classification_report(y_true, y_pred, target_names=class_names)
    
    return overall_metrics, per_class_metrics


def calculate_multilabel_metrics(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    label_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for multi-label classification.

    Args:
        y_true_list: List of true labels for each label type
        y_pred_list: List of predicted labels for each label type
        label_names: Names for each label type

    Returns:
        Dictionary of metrics for each label type
    """
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(len(y_true_list))]
        
    results = {}
    
    # Calculate metrics for each label type
    for i, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, label_names)):
        metrics = get_classification_metrics(y_true, y_pred)
        results[name] = metrics
        
    # Calculate overall accuracy (percentage of samples where all labels match)
    all_correct = np.all(np.array([y_true == y_pred for y_true, y_pred in zip(y_true_list, y_pred_list)]), axis=0)
    overall_accuracy = np.mean(all_correct)
    
    results['overall'] = {
        'exact_match_ratio': overall_accuracy
    }
    
    return results 