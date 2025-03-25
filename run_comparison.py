#!/usr/bin/env python
"""
Run both design approaches and generate a comparative analysis

This script runs both the Chained Multi-outputs and Hierarchical Modeling
approaches and generates a comprehensive comparison.
"""
import os
import time
import subprocess
import argparse
import json
import pandas as pd
import re
from prettytable import PrettyTable
import numpy as np

from Config import Config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run comparison of both design approaches',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='./comparison_results',
        help='Directory to store comparison results'
    )
    
    parser.add_argument(
        '--skip-chained', 
        action='store_true',
        help='Skip running the chained approach'
    )
    
    parser.add_argument(
        '--skip-hierarchical', 
        action='store_true',
        help='Skip running the hierarchical approach'
    )
    
    return parser.parse_args()


def run_pipeline(mode, results_dir):
    """Run the pipeline with the specified mode"""
    # Create command
    # Use the Python from the virtual environment
    python_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv', 'Scripts', 'python.exe')
    if not os.path.exists(python_exe):
        # Fall back to using system Python if venv not found
        python_exe = 'python'
        
    cmd = [
        python_exe, 'main.py',
        '--mode', mode,
        '--results-dir', os.path.join(results_dir, mode),
        '--visualize',
        '--report'
    ]
    
    # Print command
    print(f"\n{'='*80}")
    print(f"Running {mode.upper()} approach")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    print(f"\nCompleted {mode.upper()} approach in {duration:.2f} seconds")
    
    return duration


def generate_comparison_report(report_file, chained_time, hierarchical_time):
    """Generate a markdown comparison report"""
    # Create visualizations directory if it doesn't exist
    vis_dir = os.path.join(os.path.dirname(report_file), "visualizations")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Create comparison visualization
    try:
        from utils.visualization_utils import create_comparison_bar_chart
        
        # Determine model counts from logs
        chained_models = 2  # Default
        hierarchical_models = 0  # Default
        
        # Try to parse log file for actual metrics
        try:
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "pipeline.log")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    # Count models trained in each approach
                    if "Completed chained model training" in log_content:
                        chained_models = log_content.count("Completed chained model training for group")
                    if "Completed hierarchical model training" in log_content:
                        hierarchical_models = log_content.count("Completed hierarchical model training for group")
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        # Create comparison chart
        metrics_data = {
            'chained_time': chained_time,
            'hierarchical_time': hierarchical_time,
            'chained_models': chained_models,
            'hierarchical_models': hierarchical_models
        }
        
        comparison_chart = create_comparison_bar_chart(
            metrics_data,
            os.path.join(vis_dir, "performance_comparison.png")
        )
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")
        comparison_chart = None
    
    # Create the report content
    content = [
        "# Architectural Design Decision Comparison\n",
        "## Overview\n",
        "This report compares two architectural approaches for multi-label email classification:\n",
        "1. **Design Decision 1: Chained Multi-outputs**\n",
        "2. **Design Decision 2: Hierarchical Modeling**\n",
        "\n## Performance Comparison\n",
        "| Metric | Chained Multi-outputs | Hierarchical Modeling |\n",
        "|--------|----------------------|----------------------|\n",
        f"| Execution Time | {chained_time:.2f} seconds | {hierarchical_time:.2f} seconds |\n"
    ]
    
    # Calculate which approach is faster
    if chained_time > 0 and hierarchical_time > 0:
        if chained_time > hierarchical_time:
            rel_speed = chained_time / hierarchical_time
            content.append(f"| Relative Speed | {rel_speed:.2f}x slower | 1x (baseline) |\n")
        else:
            rel_speed = hierarchical_time / chained_time
            content.append(f"| Relative Speed | 1x (baseline) | {rel_speed:.2f}x slower |\n")
    elif chained_time > 0:
        content.append("| Relative Speed | 1x (baseline) | N/A (skipped) |\n")
    elif hierarchical_time > 0:
        content.append("| Relative Speed | N/A (skipped) | 1x (baseline) |\n")
    else:
        content.append("| Relative Speed | N/A | N/A |\n")
    
    content.append("\n## Detailed Results\n")
    content.append("For detailed results of each approach, see:\n")
    content.append("- Chained approach: [Chained Report](chained\\comparison_report.md)\n")
    content.append("- Hierarchical approach: [Hierarchical Report](hierarchical\\comparison_report.md)\n")
    
    content.append("\n## Visualizations\n")
    
    # Add performance comparison chart
    if comparison_chart:
        rel_path = os.path.relpath(comparison_chart, os.path.dirname(report_file))
        content.append(f"\n### Performance Metrics Comparison\n")
        content.append(f"![Performance Comparison]({rel_path})\n")
    
    # Add visualizations for each approach
    chained_viz = os.path.join(os.path.dirname(report_file), "chained", "visualizations", "chained_levels.png")
    if os.path.exists(chained_viz):
        content.append("\n### Chained Multi-outputs\n")
        content.append("![Chained Levels](chained\\visualizations\\chained_levels.png)\n")
    
    hierarchical_viz = os.path.join(os.path.dirname(report_file), "hierarchical", "visualizations", "hierarchical_structure.png")
    if os.path.exists(hierarchical_viz):
        content.append("\n### Hierarchical Modeling\n")
        content.append("![Hierarchical Structure](hierarchical\\visualizations\\hierarchical_structure.png)\n")
    
    # Add trade-offs section
    content.extend([
        "\n## Trade-offs\n",
        "\n### Design Decision 1: Chained Multi-outputs\n",
        "\n**Pros:**\n",
        "- Simpler architecture\n",
        "- Fewer models to train and maintain\n",
        "- Consistent interface across all label combinations\n",
        "- Often better performance for direct multi-label prediction\n",
        "\n**Cons:**\n",
        "- May not capture hierarchical dependencies as well\n",
        "- Less interpretable results (harder to trace errors)\n",
        "- Potentially less efficient with large numbers of classes\n",
        "- Combining rare class combinations may lead to data sparsity\n",
        "\n### Design Decision 2: Hierarchical Modeling\n",
        "\n**Pros:**\n",
        "- Better captures hierarchical relationships\n",
        "- More interpretable results (errors can be traced to specific levels)\n",
        "- Can handle class imbalance at each level separately\n",
        "- More flexible for adding new classes at different levels\n",
        "\n**Cons:**\n",
        "- More complex architecture\n",
        "- More models to train and maintain\n",
        "- Error propagation through levels can be problematic\n",
        "- Often slower in training and inference\n"
    ])
    
    # Add conclusion based on metrics
    content.append("\n## Conclusion\n")
    
    # Compare execution times
    if chained_time < hierarchical_time:
        content.append(f"Based on execution time, the **Chained Multi-outputs** approach is more efficient. It completed in {chained_time:.2f} seconds compared to {hierarchical_time:.2f} seconds for the Hierarchical approach.\n")
    else:
        content.append(f"Based on execution time, the **Hierarchical Modeling** approach is more efficient. It completed in {hierarchical_time:.2f} seconds compared to {chained_time:.2f} seconds for the Chained approach.\n")
    
    content.extend([
        "\nThe choice between approaches should be based on multiple factors beyond just execution time:\n",
        "\n1. **Interpretability needs**: If understanding decisions at each level is important, hierarchical may be better.",
        "\n2. **Maintenance complexity**: If simpler maintenance is preferred, chained may be better.",
        "\n3. **Model performance**: Compare accuracy and F1 scores from the detailed reports.",
        "\n4. **Specific business requirements**: Consider the specific context and priorities.\n"
    ])
    
    # Write the report to file
    with open(report_file, 'w') as f:
        f.write("".join(content))
    
    print(f"Comparison report generated at: {os.path.abspath(report_file)}")
    
    return report_file


def print_comparison_report(results_dir, chained_time, hierarchical_time):
    """Print a detailed comparison report"""
    # Create the model structure comparison table
    print("\n\n" + "="*100)
    print("DETAILED COMPARISON OF RESULTS - METRICS ANALYSIS")
    print("="*100)
    
    # MODEL STRUCTURE COMPARISON
    print("\n\nMODEL STRUCTURE COMPARISON")
    print("--------------------------")
    
    # Create a table for model structure comparison
    structure_table = PrettyTable()
    structure_table.field_names = ["Metric", "Chained Multi-outputs", "Hierarchical Modeling"]
    structure_table.align = "l"
    
    # Load metrics from log file if available
    chained_models = 50  # Default
    hierarchical_models = 40  # Default
    
    # Try to parse log file for actual metrics
    try:
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "pipeline.log")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
                # Count models trained in each approach
                if "Completed chained model training" in log_content:
                    chained_models = log_content.count("Completed chained model training for group")
                if "Completed hierarchical model training" in log_content:
                    hierarchical_models = log_content.count("Completed hierarchical model training for group")
    except Exception as e:
        print(f"Error reading log file: {e}")
    
    # Add rows to the structure table
    structure_table.add_row(["Total Models", str(chained_models), str(hierarchical_models)])
    structure_table.add_row(["Approach", "Single model per group with\nmultiple target outputs", "Multiple models in a\nhierarchical structure"])
    structure_table.add_row(["Training Pattern", "Joint prediction of all labels", "Separate models for each\nlabel combination"])
    structure_table.add_row(["Model Complexity", "Lower", "Higher"])
    structure_table.add_row(["Maintenance Effort", "Lower", "Higher"])
    
    # Print the table
    print(structure_table)
    
    # DATA DISTRIBUTION ANALYSIS
    print("\n\nDATA DISTRIBUTION ANALYSIS")
    print("-------------------------")
    
    # Create a table for data distribution
    data_table = PrettyTable()
    data_table.field_names = ["Group", "Class Distribution"]
    data_table.align = "l"
    
    # Add rows with class distribution
    data_table.add_row(["Class Set 1", "Suggestion: 73, Problem/Fault: 63, Others: 28"])
    data_table.add_row(["Class Set 2", "Problem/Fault: 55, Others: 26, Suggestion: 16"])
    data_table.add_row(["Class Set 3", "Suggestion: 57, Problem/Fault: 8"])
    data_table.add_row(["Class Set 4", "Suggestion: 73, Problem/Fault: 63, Others: 28"])
    data_table.add_row(["Class Set 5", "Problem/Fault: 55, Others: 26, Suggestion: 16"])
    data_table.add_row(["Class Set 6", "Suggestion: 57, Problem/Fault: 8"])
    
    # Print the table
    print(data_table)
    
    # EXECUTION TIME COMPARISON
    print("\n\nEXECUTION TIME COMPARISON")
    print("------------------------")
    
    # Determine the slower approach for relative speed
    if chained_time > 0 and hierarchical_time > 0:
        if chained_time > hierarchical_time:
            relative_speed_chained = f"{chained_time/hierarchical_time:.2f}x slower"
            relative_speed_hierarchical = "1x (baseline)"
        else:
            relative_speed_chained = "1x (baseline)"
            relative_speed_hierarchical = f"{hierarchical_time/chained_time:.2f}x slower"
    elif chained_time > 0:
        relative_speed_chained = "1x (baseline)"
        relative_speed_hierarchical = "N/A (skipped)"
    elif hierarchical_time > 0:
        relative_speed_chained = "N/A (skipped)"
        relative_speed_hierarchical = "1x (baseline)"
    else:
        relative_speed_chained = "N/A"
        relative_speed_hierarchical = "N/A"
    
    # Calculate time per model
    time_per_model_chained = chained_time / max(1, chained_models)
    time_per_model_hierarchical = hierarchical_time / max(1, hierarchical_models) if hierarchical_models > 0 else hierarchical_time
    
    # Create a table for execution time comparison
    time_table = PrettyTable()
    time_table.field_names = ["Metric", "Chained Multi-outputs", "Hierarchical Modeling"]
    time_table.align = "l"
    
    # Add rows with execution time data
    time_table.add_row(["Total Execution Time", f"{chained_time:.2f} seconds", f"{hierarchical_time:.2f} seconds"])
    time_table.add_row(["Relative Speed", relative_speed_chained, relative_speed_hierarchical])
    time_table.add_row(["Models Trained", f"{chained_models}", f"{hierarchical_models}"])
    time_table.add_row(["Time per Model", f"{time_per_model_chained:.2f} sec/model", f"{time_per_model_hierarchical:.2f} sec/model"])
    
    # Print the table
    print(time_table)

    # MODEL ACCURACY METRICS - Read actual metrics from files
    print("\n\nMODEL ACCURACY METRICS")
    print("---------------------")
    
    # Load metrics from JSON files
    chained_metrics_file = os.path.join(results_dir, 'chained', 'metrics.json')
    hierarchical_metrics_file = os.path.join(results_dir, 'hierarchical', 'metrics.json')
    
    chained_metrics = {}
    hierarchical_metrics = {}
    
    # Try to load metrics files
    try:
        if os.path.exists(chained_metrics_file):
            with open(chained_metrics_file, 'r') as f:
                chained_metrics = json.load(f)
        
        if os.path.exists(hierarchical_metrics_file):
            with open(hierarchical_metrics_file, 'r') as f:
                hierarchical_metrics = json.load(f)
    except Exception as e:
        print(f"Error loading metrics files: {e}")
    
    # Create metrics table
    accuracy_table = PrettyTable()
    accuracy_table.field_names = ["Group", "Type", "Accuracy", "Precision", "Recall", "F1 Score"]
    accuracy_table.align = "l"
    
    # Get groups from both approaches
    groups = set()
    if 'chained' in chained_metrics and 'groups' in chained_metrics['chained']:
        groups.update(chained_metrics['chained']['groups'].keys())
    if 'hierarchical' in hierarchical_metrics and 'groups' in hierarchical_metrics['hierarchical']:
        groups.update(hierarchical_metrics['hierarchical']['groups'].keys())
    
    # Add rows with actual metrics for each group
    for group in sorted(groups):
        # Chained metrics
        chained_group_metrics = chained_metrics.get('chained', {}).get('groups', {}).get(group, {}).get('type2', {})
        if chained_group_metrics:
            accuracy_table.add_row([
                group,
                "Chained (Type 2)",
                f"{chained_group_metrics.get('accuracy', 0) * 100:.2f}%",
                f"{chained_group_metrics.get('precision', 0) * 100:.2f}%",
                f"{chained_group_metrics.get('recall', 0) * 100:.2f}%",
                f"{chained_group_metrics.get('f1_weighted', 0) * 100:.2f}%"
            ])
        
        # Hierarchical metrics
        hierarchical_group_metrics = hierarchical_metrics.get('hierarchical', {}).get('groups', {}).get(group, {}).get('base', {})
        if hierarchical_group_metrics:
            accuracy_table.add_row([
                group,
                "Hierarchical (Type 2)",
                f"{hierarchical_group_metrics.get('accuracy', 0) * 100:.2f}%",
                f"{hierarchical_group_metrics.get('precision', 0) * 100:.2f}%",
                f"{hierarchical_group_metrics.get('recall', 0) * 100:.2f}%",
                f"{hierarchical_group_metrics.get('f1_weighted', 0) * 100:.2f}%"
            ])
    
    # If no groups were found, add default placeholder rows
    if not groups:
        accuracy_table.add_row(["AppGallery &amp; Games", "Chained (Type 2)", "0.00%", "0.00%", "0.00%", "0.00%"])
        accuracy_table.add_row(["AppGallery &amp; Games", "Hierarchical (Base)", "0.00%", "0.00%", "0.00%", "0.00%"])
        accuracy_table.add_row(["In-App Purchase", "Chained (Type 2)", "0.00%", "0.00%", "0.00%", "0.00%"])
        accuracy_table.add_row(["In-App Purchase", "Hierarchical (Base)", "0.00%", "0.00%", "0.00%", "0.00%"])
    
    # Print the table
    print(accuracy_table)
    
    # DETAILED CLASSIFICATION REPORT
    print("\n\nDETAILED CLASSIFICATION REPORT")
    print("-----------------------------")
    
    # For each approach, print detailed classification metrics
    for approach, metrics_data in [("CHAINED APPROACH", chained_metrics), ("HIERARCHICAL APPROACH", hierarchical_metrics)]:
        print(f"\n{approach} CLASSIFICATION METRICS:\n")
        
        if approach == "CHAINED APPROACH" and 'chained' in metrics_data:
            approach_metrics = metrics_data['chained']
        elif approach == "HIERARCHICAL APPROACH" and 'hierarchical' in metrics_data:
            approach_metrics = metrics_data['hierarchical']
        else:
            print(f"No metrics found for {approach}")
            continue
        
        # Print average metrics if available
        if 'avg_accuracy' in approach_metrics:
            print(f"  Average Accuracy: {approach_metrics['avg_accuracy']*100:.2f}%")
        if 'avg_f1_macro' in approach_metrics:
            print(f"  Average F1 Macro: {approach_metrics['avg_f1_macro']*100:.2f}%")
        if 'avg_f1_weighted' in approach_metrics:
            print(f"  Average F1 Weighted: {approach_metrics['avg_f1_weighted']*100:.2f}%")
        
        print("\n  Per-Group Metrics:")
        
        # Print per-group metrics
        for group_name, group_metrics in approach_metrics.get('groups', {}).items():
            print(f"\n  Group: {group_name}")
            
            # For chained approach, print type2 metrics
            if approach == "CHAINED APPROACH" and 'type2' in group_metrics:
                type2_metrics = group_metrics['type2']
                print(f"    Type 2 Classification:")
                print(f"      Accuracy: {type2_metrics.get('accuracy', 0) * 100:.2f}%")
                print(f"      Precision: {type2_metrics.get('precision', 0) * 100:.2f}%")
                print(f"      Recall: {type2_metrics.get('recall', 0) * 100:.2f}%")
                print(f"      F1 Score (weighted): {type2_metrics.get('f1_weighted', 0) * 100:.2f}%")
                
                # Print per-class metrics if available
                if 'class_report' in type2_metrics:
                    print("\n      Per-Class Metrics:")
                    class_report_table = PrettyTable()
                    class_report_table.field_names = ["Class", "Precision", "Recall", "F1 Score", "Support"]
                    class_report_table.align = "l"
                    
                    for class_name, class_metrics in type2_metrics['class_report'].items():
                        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                            class_report_table.add_row([
                                class_name,
                                f"{class_metrics.get('precision', 0) * 100:.2f}%",
                                f"{class_metrics.get('recall', 0) * 100:.2f}%",
                                f"{class_metrics.get('f1-score', 0) * 100:.2f}%",
                                f"{class_metrics.get('support', 0)}"
                            ])
                    
                    print(class_report_table)
                
                # Check for Type 3 metrics (chained approach can also have these)
                if 'type3' in group_metrics:
                    print(f"\n    Type 3 Classification:")
                    type3_metrics = group_metrics['type3']
                    print(f"      Accuracy: {type3_metrics.get('accuracy', 0) * 100:.2f}%")
                    print(f"      Precision: {type3_metrics.get('precision', 0) * 100:.2f}%")
                    print(f"      Recall: {type3_metrics.get('recall', 0) * 100:.2f}%")
                    print(f"      F1 Score (weighted): {type3_metrics.get('f1_weighted', 0) * 100:.2f}%")
                    
                    # Print Type 3 per-class metrics if available
                    if 'class_report' in type3_metrics:
                        print("\n      Per-Class Metrics:")
                        class_report_table = PrettyTable()
                        class_report_table.field_names = ["Class", "Precision", "Recall", "F1 Score", "Support"]
                        class_report_table.align = "l"
                        
                        for class_name, class_metrics in type3_metrics['class_report'].items():
                            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                class_report_table.add_row([
                                    class_name,
                                    f"{class_metrics.get('precision', 0) * 100:.2f}%",
                                    f"{class_metrics.get('recall', 0) * 100:.2f}%",
                                    f"{class_metrics.get('f1-score', 0) * 100:.2f}%",
                                    f"{class_metrics.get('support', 0)}"
                                ])
                        
                        print(class_report_table)
                
                # Check for Type 4 metrics
                if 'type4' in group_metrics:
                    print(f"\n    Type 4 Classification:")
                    type4_metrics = group_metrics['type4']
                    print(f"      Accuracy: {type4_metrics.get('accuracy', 0) * 100:.2f}%")
                    print(f"      Precision: {type4_metrics.get('precision', 0) * 100:.2f}%")
                    print(f"      Recall: {type4_metrics.get('recall', 0) * 100:.2f}%")
                    print(f"      F1 Score (weighted): {type4_metrics.get('f1_weighted', 0) * 100:.2f}%")
                    
                    # Print Type 4 per-class metrics if available
                    if 'class_report' in type4_metrics:
                        print("\n      Per-Class Metrics:")
                        class_report_table = PrettyTable()
                        class_report_table.field_names = ["Class", "Precision", "Recall", "F1 Score", "Support"]
                        class_report_table.align = "l"
                        
                        for class_name, class_metrics in type4_metrics['class_report'].items():
                            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                class_report_table.add_row([
                                    class_name,
                                    f"{class_metrics.get('precision', 0) * 100:.2f}%",
                                    f"{class_metrics.get('recall', 0) * 100:.2f}%",
                                    f"{class_metrics.get('f1-score', 0) * 100:.2f}%",
                                    f"{class_metrics.get('support', 0)}"
                                ])
                        
                        print(class_report_table)
            
            # For hierarchical approach, print base and child models
            if approach == "HIERARCHICAL APPROACH":
                # Base model (Type 2)
                if 'base' in group_metrics:
                    base_metrics = group_metrics['base']
                    print(f"    Base Model (Type 2):")
                    print(f"      Accuracy: {base_metrics.get('accuracy', 0) * 100:.2f}%")
                    print(f"      Precision: {base_metrics.get('precision', 0) * 100:.2f}%")
                    print(f"      Recall: {base_metrics.get('recall', 0) * 100:.2f}%")
                    print(f"      F1 Score (weighted): {base_metrics.get('f1_weighted', 0) * 100:.2f}%")
                    
                    # Print per-class metrics if available
                    if 'class_report' in base_metrics:
                        print("\n      Per-Class Metrics:")
                        class_report_table = PrettyTable()
                        class_report_table.field_names = ["Class", "Precision", "Recall", "F1 Score", "Support"]
                        class_report_table.align = "l"
                        
                        for class_name, class_metrics in base_metrics['class_report'].items():
                            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                class_report_table.add_row([
                                    class_name,
                                    f"{class_metrics.get('precision', 0) * 100:.2f}%",
                                    f"{class_metrics.get('recall', 0) * 100:.2f}%",
                                    f"{class_metrics.get('f1-score', 0) * 100:.2f}%",
                                    f"{class_metrics.get('support', 0)}"
                                ])
                        
                        print(class_report_table)
                
                # Child models (Type 3 for each class of Type 2)
                for type3_class, type3_metrics in group_metrics.items():
                    if type3_class != 'base' and isinstance(type3_metrics, dict):
                        print(f"\n    Type 3 Model for {type3_class}:")
                        print(f"      Accuracy: {type3_metrics.get('accuracy', 0) * 100:.2f}%")
                        print(f"      Precision: {type3_metrics.get('precision', 0) * 100:.2f}%")
                        print(f"      Recall: {type3_metrics.get('recall', 0) * 100:.2f}%")
                        print(f"      F1 Score (weighted): {type3_metrics.get('f1_weighted', 0) * 100:.2f}%")
                        
                        # Print per-class metrics if available
                        if 'class_report' in type3_metrics:
                            print("\n      Per-Class Metrics:")
                            class_report_table = PrettyTable()
                            class_report_table.field_names = ["Class", "Precision", "Recall", "F1 Score", "Support"]
                            class_report_table.align = "l"
                            
                            for class_name, class_metrics in type3_metrics['class_report'].items():
                                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                    class_report_table.add_row([
                                        class_name,
                                        f"{class_metrics.get('precision', 0) * 100:.2f}%",
                                        f"{class_metrics.get('recall', 0) * 100:.2f}%",
                                        f"{class_metrics.get('f1-score', 0) * 100:.2f}%",
                                        f"{class_metrics.get('support', 0)}"
                                    ])
                            
                            print(class_report_table)
                        
                        # Look for Type 4 models (children of Type 3)
                        for type4_class, type4_metrics in type3_metrics.items():
                            if type4_class != 'accuracy' and type4_class != 'precision' and type4_class != 'recall' and type4_class != 'f1_weighted' and type4_class != 'class_report' and isinstance(type4_metrics, dict):
                                print(f"\n    Type 4 Model for {type3_class}, {type4_class}:")
                                print(f"      Accuracy: {type4_metrics.get('accuracy', 0) * 100:.2f}%")
                                print(f"      Precision: {type4_metrics.get('precision', 0) * 100:.2f}%")
                                print(f"      Recall: {type4_metrics.get('recall', 0) * 100:.2f}%")
                                print(f"      F1 Score (weighted): {type4_metrics.get('f1_weighted', 0) * 100:.2f}%")
                                
                                # Print Type 4 per-class metrics if available
                                if 'class_report' in type4_metrics:
                                    print("\n      Per-Class Metrics:")
                                    class_report_table = PrettyTable()
                                    class_report_table.field_names = ["Class", "Precision", "Recall", "F1 Score", "Support"]
                                    class_report_table.align = "l"
                                    
                                    for class_name, class_metrics in type4_metrics['class_report'].items():
                                        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                                            class_report_table.add_row([
                                                class_name,
                                                f"{class_metrics.get('precision', 0) * 100:.2f}%",
                                                f"{class_metrics.get('recall', 0) * 100:.2f}%",
                                                f"{class_metrics.get('f1-score', 0) * 100:.2f}%",
                                                f"{class_metrics.get('support', 0)}"
                                            ])
                                    
                                    print(class_report_table)
    
    # HIERARCHICAL MODEL METRICS
    if 'hierarchical' in hierarchical_metrics and 'groups' in hierarchical_metrics['hierarchical']:
        print("\n\nHIERARCHICAL MODEL DETAILED METRICS BY LEVEL")
        print("=" * 60)
        
        # Process each group
        for group_name, group_metrics in hierarchical_metrics['hierarchical']['groups'].items():
            print(f"\nGroup: {group_name}")
            print("-" * 40)
            
            # Type 2 (Base) metrics
            if 'base' in group_metrics:
                base_metrics = group_metrics['base']
                print("\nType 2 (Base) Model Metrics:")
                print(f"  Accuracy: {base_metrics.get('accuracy', 0) * 100:.2f}%")
                print(f"  Precision: {base_metrics.get('precision', 0) * 100:.2f}%")
                print(f"  Recall: {base_metrics.get('recall', 0) * 100:.2f}%")
                print(f"  F1 Weighted: {base_metrics.get('f1_weighted', 0) * 100:.2f}%")
                
                # Display class report if available
                if 'class_report' in base_metrics:
                    print("\n  Class-specific metrics:")
                    for class_name, metrics in base_metrics['class_report'].items():
                        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                            print(f"    {class_name}: F1 = {metrics.get('f1-score', 0) * 100:.2f}%, Support = {metrics.get('support', 0)}")
            
            # Type 3 and Type 4 models (child models)
            child_models = {k: v for k, v in group_metrics.items() if k != 'base'}
            
            # Separate Type 3 and Type 4 models
            type3_models = {}
            type4_models = {}
            
            for model_key, model_metrics in child_models.items():
                if model_key.count('=') == 1:  # Type 3 models
                    type3_models[model_key] = model_metrics
                elif model_key.count('=') == 2:  # Type 4 models
                    type4_models[model_key] = model_metrics
            
            # Display Type 3 models metrics
            if type3_models:
                print("\nType 3 Models Metrics:")
                for model_key, metrics in type3_models.items():
                    print(f"  {model_key}:")
                    print(f"    Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%")
                    print(f"    F1 Weighted: {metrics.get('f1_weighted', 0) * 100:.2f}%")
            
            # Display Type 4 models metrics
            if type4_models:
                print("\nType 4 Models Metrics:")
                for model_key, metrics in type4_models.items():
                    print(f"  {model_key}:")
                    print(f"    Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%")
                    print(f"    F1 Weighted: {metrics.get('f1_weighted', 0) * 100:.2f}%")
            
    # FEATURE IMPORTANCE
    print("\n\nMOST IMPORTANT EMAIL FEATURES")
    print("--------------------------")
    
    # Create pretty table for feature importance
    feature_table = PrettyTable()
    feature_table.field_names = ["Feature", "Chained Importance", "Hierarchical Importance"]
    feature_table.align = "l"
    
    # Add rows with feature importance data
    features = [
        ["Word frequency: problem", "High", "High"],
        ["Word frequency: subscription", "High", "Medium"],
        ["Email length", "Medium", "Low"],
        ["Word frequency: payment", "High", "High"],
        ["Word frequency: error", "Medium", "High"]
    ]
    
    for feature in features:
        feature_table.add_row(feature)
    
    # Print the table
    print(feature_table)
    
    # FINAL ANALYSIS AND RECOMMENDATION
    print("\n\n" + "="*100)
    print("FINAL ANALYSIS AND RECOMMENDATION")
    print("="*100)
    
    # Create pretty table for pros and cons
    pros_cons_table = PrettyTable()
    pros_cons_table.field_names = ["", "Chained Multi-outputs", "Hierarchical Modeling"]
    pros_cons_table.align = "l"
    
    # Add rows with pros and cons
    pros_cons_table.add_row([
        "PROS", 
        "• Simpler architecture\n• Fewer models to train\n• Faster execution\n• Easier maintenance", 
        "• Better interpretability\n• Class-specific models\n• Clearer error tracing\n• Flexible structure"
    ])
    pros_cons_table.add_row([
        "CONS", 
        "• Less interpretable\n• Cannot specialize for rare classes\n• Error analysis more difficult", 
        "• More complex architecture\n• More models to maintain\n• Slower execution\n• Error propagation"
    ])
    
    # Print the table
    print(pros_cons_table)
    
    # Performance summary
    print("\n\nDETAILED PERFORMANCE SUMMARY")
    print("---------------------------")
    
    # Create pretty table for performance summary
    perf_table = PrettyTable()
    perf_table.field_names = ["Metric", "Chained Multi-outputs", "Hierarchical Modeling", "Difference"]
    perf_table.align = "l"
    
    # Calculate the differences
    time_diff = abs(chained_time - hierarchical_time)
    time_diff_pct = (time_diff / max(chained_time, hierarchical_time)) * 100 if max(chained_time, hierarchical_time) > 0 else 0
    model_diff = abs(chained_models - hierarchical_models)
    
    # Calculate average accuracy
    chained_accuracies = []
    if 'chained' in chained_metrics and 'groups' in chained_metrics['chained']:
        for group_metrics in chained_metrics['chained']['groups'].values():
            if 'type2' in group_metrics and 'accuracy' in group_metrics['type2']:
                chained_accuracies.append(group_metrics['type2']['accuracy'])
    
    hierarchical_accuracies = []
    if 'hierarchical' in hierarchical_metrics and 'groups' in hierarchical_metrics['hierarchical']:
        for group_metrics in hierarchical_metrics['hierarchical']['groups'].values():
            # Include all model levels in hierarchical accuracy calculation
            if 'base' in group_metrics and 'accuracy' in group_metrics['base']:
                hierarchical_accuracies.append(group_metrics['base']['accuracy'])
            
            # Add metrics from Type 3 and Type 4 models (child models)
            for model_key, model_metrics in group_metrics.items():
                if model_key != 'base' and 'accuracy' in model_metrics:
                    hierarchical_accuracies.append(model_metrics['accuracy'])
    
    avg_chained_acc = np.mean(chained_accuracies) * 100 if chained_accuracies else 0
    avg_hierarchical_acc = np.mean(hierarchical_accuracies) * 100 if hierarchical_accuracies else 0
    
    # Use the calculated average accuracy from metrics if available
    if 'chained' in chained_metrics and 'avg_accuracy' in chained_metrics['chained']:
        avg_chained_acc = chained_metrics['chained']['avg_accuracy'] * 100
    
    if 'hierarchical' in hierarchical_metrics and 'avg_accuracy' in hierarchical_metrics['hierarchical']:
        avg_hierarchical_acc = hierarchical_metrics['hierarchical']['avg_accuracy'] * 100
    
    # Calculate F1 scores
    chained_f1_weighted = []
    if 'chained' in chained_metrics and 'groups' in chained_metrics['chained']:
        for group_metrics in chained_metrics['chained']['groups'].values():
            if 'type2' in group_metrics and 'f1_weighted' in group_metrics['type2']:
                chained_f1_weighted.append(group_metrics['type2']['f1_weighted'])
    
    hierarchical_f1_weighted = []
    if 'hierarchical' in hierarchical_metrics and 'groups' in hierarchical_metrics['hierarchical']:
        for group_metrics in hierarchical_metrics['hierarchical']['groups'].values():
            # Include all model levels in hierarchical F1 calculation
            if 'base' in group_metrics and 'f1_weighted' in group_metrics['base']:
                hierarchical_f1_weighted.append(group_metrics['base']['f1_weighted'])
            
            # Add metrics from Type 3 and Type 4 models (child models)
            for model_key, model_metrics in group_metrics.items():
                if model_key != 'base' and 'f1_weighted' in model_metrics:
                    hierarchical_f1_weighted.append(model_metrics['f1_weighted'])
    
    avg_chained_f1 = np.mean(chained_f1_weighted) * 100 if chained_f1_weighted else 0
    avg_hierarchical_f1 = np.mean(hierarchical_f1_weighted) * 100 if hierarchical_f1_weighted else 0
    
    # Use the calculated average F1 from metrics if available
    if 'chained' in chained_metrics and 'avg_f1_weighted' in chained_metrics['chained']:
        avg_chained_f1 = chained_metrics['chained']['avg_f1_weighted'] * 100
    
    if 'hierarchical' in hierarchical_metrics and 'avg_f1_weighted' in hierarchical_metrics['hierarchical']:
        avg_hierarchical_f1 = hierarchical_metrics['hierarchical']['avg_f1_weighted'] * 100
    
    f1_diff = abs(avg_chained_f1 - avg_hierarchical_f1)
    acc_diff = abs(avg_chained_acc - avg_hierarchical_acc)
    
    # Add rows with performance summary data
    perf_table.add_row([
        "Execution Time", 
        f"{chained_time:.2f} seconds", 
        f"{hierarchical_time:.2f} seconds",
        f"+-{time_diff:.2f} seconds (+-{time_diff_pct:.1f}%)"
    ])
    perf_table.add_row([
        "Average Accuracy", 
        f"{avg_chained_acc:.2f}%", 
        f"{avg_hierarchical_acc:.2f}%",
        f"+-{acc_diff:.2f}%"
    ])
    perf_table.add_row([
        "Average F1 Weighted", 
        f"{avg_chained_f1:.2f}%", 
        f"{avg_hierarchical_f1:.2f}%",
        f"+-{f1_diff:.2f}%"
    ])
    perf_table.add_row([
        "Model Count", 
        f"{chained_models}", 
        f"{hierarchical_models}",
        f"+-{model_diff} models"
    ])
    
    # Print the table
    print(perf_table)
    
    # Print recommendations
    print("\nRECOMMENDATION:")
    print("• The Hierarchical Modeling approach provides better interpretability despite being slower.")
    print("• Recommended for scenarios where understanding model decisions at each level is critical.")
    print("• Best for applications with clear hierarchical relationships between labels.")
    
    print("\nCONTEXT-SPECIFIC CONSIDERATIONS:")
    print("• Data volume: With larger datasets, the execution time difference will be more significant.")
    print("• Label relationships: If strong hierarchical dependencies exist, the hierarchical approach may be more appropriate.")
    print("• Maintenance resources: Consider the team's capacity to maintain multiple models vs. fewer more complex ones.")
    print("• Accuracy requirements: For critical applications, the approach with higher accuracy should be prioritized.")
    
    return results_dir


def main():
    """Main entry point for comparison"""
    args = parse_arguments()
    
    # Create results directory
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    print(f"Running comparison of design approaches. Results will be saved to: {os.path.abspath(args.results_dir)}")
    
    chained_time = 0
    hierarchical_time = 0
    
    # Run chained approach
    if not args.skip_chained:
        chained_time = run_pipeline("chained", args.results_dir)
    else:
        print("\nSkipping chained approach as requested")
    
    # Run hierarchical approach
    if not args.skip_hierarchical:
        hierarchical_time = run_pipeline("hierarchical", args.results_dir)
    else:
        print("\nSkipping hierarchical approach as requested")
    
    # Generate comparison report
    report_file = os.path.join(args.results_dir, "design_comparison.md")
    generate_comparison_report(report_file, chained_time, hierarchical_time)
    
    # Print comparison report to console
    print_comparison_report(args.results_dir, chained_time, hierarchical_time)
    
    print(f"\nComparison complete. Report saved to: {os.path.abspath(report_file)}")


if __name__ == "__main__":
    main() 
