"""
Visualization utilities for email classification
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List, Optional, Tuple, Union, Dict, Any


def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    normalize: bool = False
) -> plt.Figure:
    """
    Create and optionally save a confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names (if None, unique values from y_true will be used)
        title: Title for the plot
        output_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size
        cmap: Colormap for the heatmap
        normalize: Whether to normalize the confusion matrix

    Returns:
        Figure object
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Get unique labels if not provided
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred])).tolist()
    
    # Create figure
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd', 
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    # Set labels and title
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout to ensure all elements are visible
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_feature_importance(
    feature_importance: np.ndarray,
    feature_names: List[str] = None,
    top_n: int = 20,
    title: str = "Feature Importance",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create and optionally save a feature importance plot.

    Args:
        feature_importance: Array of feature importance values
        feature_names: Names of features (if None, indexes will be used)
        top_n: Number of top features to display
        title: Title for the plot
        output_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size

    Returns:
        Figure object
    """
    # If feature names are not provided, use indices
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
    
    # Create dataframe for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Take top N features
    if top_n > 0 and top_n < len(importance_df):
        importance_df = importance_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create horizontal bar chart
    sns.barplot(
        data=importance_df,
        y='Feature',
        x='Importance',
        palette='viridis'
    )
    
    # Set labels and title
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    
    # Add grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = "accuracy",
    title: str = "Model Comparison",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a comparison plot for a specific metric across different models.

    Args:
        metrics: Dictionary of model metrics in format {model_name: {metric_name: value}}
        metric_name: Name of the metric to compare
        title: Title for the plot
        output_path: Path to save the figure (if None, figure is not saved)
        figsize: Figure size

    Returns:
        Figure object
    """
    # Extract model names and metric values
    model_names = []
    metric_values = []
    
    for model_name, model_metrics in metrics.items():
        if metric_name in model_metrics:
            model_names.append(model_name)
            metric_values.append(model_metrics[metric_name])
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create bar plot
    bars = plt.bar(model_names, metric_values, color=sns.color_palette("viridis", len(model_names)))
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.4f}',
            ha='center',
            va='bottom'
        )
    
    # Set labels and title
    plt.xlabel('Model')
    plt.ylabel(metric_name.capitalize())
    plt.title(title)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability if many models
    if len(model_names) > 4:
        plt.xticks(rotation=45, ha='right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def create_comparison_bar_chart(metrics_data, output_file):
    """
    Create a bar chart comparing key performance metrics between approaches
    
    Args:
        metrics_data (dict): Dictionary containing metrics for both approaches
        output_file (str): Path to save the output image
    """
    try:
        # Set up metrics to compare
        metrics = [
            ('Execution Time (sec)', metrics_data.get('chained_time', 0), metrics_data.get('hierarchical_time', 0)),
            ('Models Count', metrics_data.get('chained_models', 0), metrics_data.get('hierarchical_models', 0)),
            ('Time per Model (sec)', 
             metrics_data.get('chained_time', 0) / max(1, metrics_data.get('chained_models', 1)),
             metrics_data.get('hierarchical_time', 0) / max(1, metrics_data.get('hierarchical_models', 1)))
        ]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set width of bars
        bar_width = 0.35
        
        # Set positions of the bars on X axis
        r1 = np.arange(len(metrics))
        r2 = [x + bar_width for x in r1]
        
        # Create bars
        ax.bar(r1, [m[1] for m in metrics], width=bar_width, label='Chained', color='skyblue')
        ax.bar(r2, [m[2] for m in metrics], width=bar_width, label='Hierarchical', color='lightcoral')
        
        # Add labels and title
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance Comparison: Chained vs. Hierarchical')
        ax.set_xticks([r + bar_width/2 for r in range(len(metrics))])
        ax.set_xticklabels([m[0] for m in metrics])
        
        # Add legend
        ax.legend()
        
        # Add value labels on top of bars
        for i, rect in enumerate(ax.patches):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', rotation=0)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to create comparison chart: {str(e)}")
        # Create a placeholder image
        create_placeholder_image(output_file, f"Error creating comparison chart: {str(e)}")
        return output_file 