"""
Base pipeline implementation with decorators for pipeline stages
"""
from abc import ABC, abstractmethod
from functools import wraps
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineContext:
    """Context object passed between pipeline stages"""
    def __init__(self):
        self._data = {}
        self.metrics = {}
        self.start_time = time.time()
        
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context"""
        self._data[key] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context"""
        return self._data.get(key, default)
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple values in the context"""
        self._data.update(data)
    
    def keys(self) -> List[str]:
        """Get all keys in the context"""
        return list(self._data.keys())
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric to track pipeline performance"""
        self.metrics[name] = value
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since pipeline started"""
        return time.time() - self.start_time
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the context"""
        return key in self._data


class PipelineStage(ABC):
    """Base class for all pipeline stages"""
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.execution_time = 0
        
    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Process the context and return updated context"""
        pass
    
    def __call__(self, context: PipelineContext) -> PipelineContext:
        """Make the stage callable for easier pipeline composition"""
        logger.info(f"Executing stage: {self.name}")
        start_time = time.time()
        result = self.process(context)
        self.execution_time = time.time() - start_time
        logger.info(f"Completed stage: {self.name} in {self.execution_time:.2f} seconds")
        return result


def pipeline_stage(requires: List[str] = None, provides: List[str] = None):
    """Decorator for pipeline stages to document and validate input/output"""
    def decorator(cls):
        original_process = cls.process
        
        @wraps(original_process)
        def wrapped_process(self, context: PipelineContext) -> PipelineContext:
            # Validate required context keys
            if requires:
                missing = [key for key in requires if key not in context]
                if missing:
                    raise ValueError(f"Pipeline stage {self.name} missing required context keys: {missing}")
            
            # Execute the original process method
            result = original_process(self, context)
            
            # Validate provided context keys
            if provides:
                missing = [key for key in provides if key not in context]
                if missing:
                    logger.warning(f"Pipeline stage {self.name} did not provide expected keys: {missing}")
            
            return result
        
        cls.process = wrapped_process
        cls._requires = requires or []
        cls._provides = provides or []
        return cls
    
    return decorator


class Pipeline:
    """Main pipeline class that executes stages in sequence"""
    def __init__(self, stages: List[PipelineStage] = None, name: str = "Pipeline"):
        self.stages = stages or []
        self.name = name
        self.context = PipelineContext()
        
    def add_stage(self, stage: PipelineStage) -> 'Pipeline':
        """Add a stage to the pipeline"""
        self.stages.append(stage)
        return self
    
    def run(self) -> PipelineContext:
        """Run all stages in the pipeline"""
        logger.info(f"Starting pipeline: {self.name}")
        start_time = time.time()
        
        for stage in self.stages:
            self.context = stage(self.context)
            
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        
        # Add total pipeline execution time to metrics
        self.context.add_metric("total_execution_time", total_time)
        
        return self.context
    
    def __str__(self) -> str:
        """String representation of the pipeline"""
        stages_str = '\n  '.join([s.name for s in self.stages])
        return f"Pipeline: {self.name}\nStages:\n  {stages_str}"
    
    @property
    def description(self) -> Dict[str, Any]:
        """Get a description of the pipeline structure"""
        return {
            "name": self.name,
            "stages": [
                {
                    "name": stage.name,
                    "requires": getattr(stage, "_requires", []),
                    "provides": getattr(stage, "_provides", [])
                }
                for stage in self.stages
            ]
        } 