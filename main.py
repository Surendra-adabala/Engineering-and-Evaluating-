#!/usr/bin/env python
"""
Multi-label Email Classification using Pipeline-Based Approach with Decorators

This script implements both chained multi-output and hierarchical modeling
approaches using a pipeline-based architecture with decorators.
"""
import argparse
import time
import os
import logging

from modelling import (
    Pipeline,
    LoadDataStage,
    PreprocessDataStage,
    EmbeddingsStage,
    DataPreparationStage,
    SingleModelStage,
    ChainedModelStage,
    HierarchicalModelStage,
    EvaluationStage,
    ReportingStage,
    VisualizationStage
)
from utils.logging_utils import setup_logging
from Config import Config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Email Classification with Pipeline-Based Architecture',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default=Config.PIPELINE_MODE,
        choices=['chained', 'hierarchical', 'auto'],
        help='Mode to run: chained (Design Decision 1), hierarchical (Design Decision 2), or auto (uses Config)'
    )
    
    parser.add_argument(
        '--log-file', 
        type=str, 
        default=os.path.join(Config.LOGS_DIR, 'pipeline.log'),
        help='Path to log file'
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default=Config.RESULTS_DIR,
        help='Directory to store results'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true', 
        default=Config.VISUALIZE_RESULTS,
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--report', 
        action='store_true', 
        default=Config.GENERATE_REPORTS,
        help='Generate detailed reports'
    )
    
    return parser.parse_args()


def create_pipeline(args):
    """Create the appropriate pipeline based on arguments"""
    # Set up logging
    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setup_logging(log_file=args.log_file)
    
    # Create results directory if it doesn't exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Set chained mode based on arguments
    if args.mode != 'auto':
        Config.CHAINED_MODE = (args.mode == 'chained')
    
    # Create the base pipeline
    pipeline = Pipeline(name="Email Classification Pipeline")
    
    # Add data loading and preprocessing stages
    pipeline.add_stage(LoadDataStage())
    pipeline.add_stage(PreprocessDataStage())
    pipeline.add_stage(EmbeddingsStage(**Config.TFIDF_PARAMS))
    pipeline.add_stage(DataPreparationStage())
    
    # Add modeling stages based on mode
    if Config.CHAINED_MODE:
        pipeline.add_stage(ChainedModelStage(
            n_estimators=Config.RF_PARAMS["chained"]["n_estimators"]
        ))
    else:
        pipeline.add_stage(HierarchicalModelStage(
            n_estimators=Config.RF_PARAMS["hierarchical"]["base"]["n_estimators"],
            child_estimators=Config.RF_PARAMS["hierarchical"]["child"]["n_estimators"]
        ))
    
    # Add evaluation stages
    pipeline.add_stage(EvaluationStage(results_dir=args.results_dir))
    
    # Add reporting stage if requested
    if args.report:
        pipeline.add_stage(ReportingStage(results_dir=args.results_dir))
    
    # Add visualization stage if requested
    if args.visualize:
        pipeline.add_stage(VisualizationStage(results_dir=args.results_dir))
    
    return pipeline


def main():
    """Main entry point"""
    args = parse_arguments()
    
    print(f"Starting email classification using {'Chained Multi-outputs' if Config.CHAINED_MODE else 'Hierarchical Modeling'}")
    print(f"Results will be saved to: {os.path.abspath(args.results_dir)}")
    
    start_time = time.time()
    
    # Create and run the pipeline
    pipeline = create_pipeline(args)
    context = pipeline.run()
    
    # Print summary of execution
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Pipeline completed with {len(context.metrics)} metrics tracked")
    
    # Print path to report if generated
    if args.report and "report_file" in context:
        print(f"Detailed report available at: {context.get('report_file')}")
    
    # Print path to visualizations if generated
    if args.visualize and "visualization_files" in context:
        viz_files = context.get("visualization_files")
        if viz_files:
            print(f"Visualizations available in: {os.path.dirname(viz_files[0])}")


if __name__ == "__main__":
    main() 
