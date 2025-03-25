"""
Preprocessing pipeline stages for email classification
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import re
import sys
import os

# Add the root directory to the Python path to allow importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_pipeline import PipelineStage, PipelineContext, pipeline_stage
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the original preprocessing functions to reuse
from preprocess import get_input_data, de_duplication, noise_remover
from Config import Config

# Import the embeddings functions to reuse
from embeddings import get_tfidf_embd

logger = logging.getLogger(__name__)

@pipeline_stage(
    provides=["raw_data"]
)
class LoadDataStage(PipelineStage):
    """Stage to load the raw data from CSV files"""
    
    def __init__(self, data_paths: List[str] = None, name: str = "LoadDataStage"):
        super().__init__(name)
        self.data_paths = data_paths or [
            "data/AppGallery.csv",
            "data/Purchasing.csv"
        ]
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Load the data from CSV files"""
        logger.info(f"Loading data from {len(self.data_paths)} files")
        
        # Reuse the get_input_data function but with customization
        if not self.data_paths:
            df = get_input_data()  # Use existing function if no paths provided
        else:
            # Load and process each file
            dfs = []
            for path in self.data_paths:
                logger.info(f"Loading data from {path}")
                df = pd.read_csv(path, skipinitialspace=True)
                
                # Rename columns to standardized format
                if 'Type 1' in df.columns:
                    df.rename(columns={
                        'Type 1': 'y1', 
                        'Type 2': 'y2', 
                        'Type 3': 'y3', 
                        'Type 4': 'y4'
                    }, inplace=True)
                
                dfs.append(df)
            
            # Concatenate all dataframes
            df = pd.concat(dfs, ignore_index=True)
            
            # Process data as in original get_input_data
            df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
            df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
            df["y"] = df[Config.CLASS_COL]
            df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
        
        # Store in context and log results
        context.set("raw_data", df)
        logger.info(f"Loaded {len(df)} rows of data")
        
        return context


@pipeline_stage(
    requires=["raw_data"],
    provides=["preprocessed_data"]
)
class PreprocessDataStage(PipelineStage):
    """Stage to preprocess the data"""
    
    def __init__(self, name: str = "PreprocessDataStage"):
        super().__init__(name)
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Preprocess the data"""
        df = context.get("raw_data")
        logger.info(f"Preprocessing {len(df)} rows of data")
        
        # Track initial row count for metrics
        initial_row_count = len(df)
        
        # Apply deduplication
        logger.info("Applying deduplication")
        df = de_duplication(df)
        
        # Apply noise removal
        logger.info("Applying noise removal")
        df = noise_remover(df)
        
        # Store in context and log results
        context.set("preprocessed_data", df)
        
        # Calculate and log data reduction metrics
        final_row_count = len(df)
        reduction_pct = (initial_row_count - final_row_count) / initial_row_count * 100
        context.add_metric("initial_row_count", initial_row_count)
        context.add_metric("final_row_count", final_row_count)
        context.add_metric("row_reduction_percent", reduction_pct)
        
        logger.info(f"Preprocessing complete. Rows reduced from {initial_row_count} to {final_row_count} ({reduction_pct:.2f}%)")
        
        return context


@pipeline_stage(
    requires=["preprocessed_data"],
    provides=["embeddings", "embedding_model"]
)
class EmbeddingsStage(PipelineStage):
    """Stage to generate embeddings from preprocessed data"""
    
    def __init__(self, 
                 max_features: int = 2000, 
                 min_df: int = 4, 
                 max_df: float = 0.9,
                 name: str = "EmbeddingsStage"):
        super().__init__(name)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
    def process(self, context: PipelineContext) -> PipelineContext:
        """Generate embeddings from the preprocessed data"""
        df = context.get("preprocessed_data")
        logger.info(f"Generating embeddings for {len(df)} documents")
        
        # Initialize the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=self.max_features, 
            min_df=self.min_df, 
            max_df=self.max_df
        )
        
        # Combine text fields and transform
        text_data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
        X = vectorizer.fit_transform(text_data).toarray()
        
        # Store embeddings and vectorizer in context
        context.set("embeddings", X)
        context.set("embedding_model", vectorizer)
        
        # Log embedding dimensions and vocabulary size
        logger.info(f"Generated embeddings with shape: {X.shape}")
        logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # Store metrics
        context.add_metric("embedding_dimensions", X.shape[1])
        context.add_metric("vocabulary_size", len(vectorizer.vocabulary_))
        
        return context 