"""
Pipeline package for email classification
"""
import sys
import os

# Add the root directory to the Python path to allow importing modules from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_pipeline import Pipeline, PipelineStage, pipeline_stage
from .preprocessing_pipeline import (
    LoadDataStage,
    PreprocessDataStage,
    EmbeddingsStage
)
from .modeling_pipeline import (
    DataPreparationStage,
    SingleModelStage,
    ChainedModelStage,
    HierarchicalModelStage
)
from .evaluation_pipeline import (
    EvaluationStage,
    ReportingStage,
    VisualizationStage
)

__all__ = [
    'Pipeline',
    'PipelineStage',
    'pipeline_stage',
    'LoadDataStage',
    'PreprocessDataStage',
    'EmbeddingsStage',
    'DataPreparationStage',
    'SingleModelStage',
    'ChainedModelStage',
    'HierarchicalModelStage',
    'EvaluationStage',
    'ReportingStage',
    'VisualizationStage'
] 