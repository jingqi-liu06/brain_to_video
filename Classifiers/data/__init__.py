"""EEG Data Processing Module.

This module provides a unified data processing pipeline for EEG classification,
ensuring consistency between training and inference.
"""

from .dataset import EEGDataset, EEGPreprocessor

__all__ = ["EEGDataset", "EEGPreprocessor"]
