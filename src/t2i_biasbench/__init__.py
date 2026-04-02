"""T2I-BiasBench: modular fairness and bias evaluation for text-to-image captions."""

from .config import DatasetConfig, StudyConfig, load_dataset_config, load_study_config
from .evaluate import evaluate_model
from .io_utils import load_caption_dataset

__all__ = [
    "DatasetConfig",
    "StudyConfig",
    "load_dataset_config",
    "load_study_config",
    "evaluate_model",
    "load_caption_dataset",
]
