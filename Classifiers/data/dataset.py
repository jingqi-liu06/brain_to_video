"""EEG Dataset and Preprocessor classes.

This module implements a modular data processing pipeline that strictly follows
the logic of train_classifier_mono.py and multi_inference.py.

Key processing order:
1. Feature extraction on ORIGINAL (un-normalized) data
2. Compute/load raw normalization statistics (mean, std per channel)
3. Normalize raw data
4. Fit/apply StandardScaler on features
5. Concatenate normalized raw + scaled features

Usage:
------
Training:
    preprocessor = EEGPreprocessor(model_type='glmnet', fs=200, win_sec=1.0)
    preprocessor.fit(X_train)  # Compute stats on training data only
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    preprocessor.save(checkpoint_dir)

Inference:
    preprocessor = EEGPreprocessor(model_type='glmnet', fs=200, win_sec=1.0)
    preprocessor.load(checkpoint_dir)  # Load saved stats
    X_test_processed = preprocessor.transform(X_test)
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Classifiers.modules.models import mlpnet


class EEGPreprocessor:
    """EEG data preprocessor that handles feature extraction, normalization, and scaling.
    
    This class ensures consistent data processing between training and inference
    by encapsulating all preprocessing logic in a single, reusable component.
    
    Attributes:
        model_type (str): Type of model ('glmnet', 'eegnet', 'deepnet').
        fs (int): Sampling frequency in Hz.
        win_sec (float): Window duration in seconds.
        raw_mean (np.ndarray): Per-channel mean from training data.
        raw_std (np.ndarray): Per-channel std from training data.
        scaler (StandardScaler): Feature scaler fitted on training data.
    """
    
    def __init__(self, model_type: str = 'glmnet', fs: int = 200, win_sec: float = 1.0):
        """Initialize the preprocessor.
        
        Args:
            model_type: Model architecture type. 'glmnet' requires feature extraction.
            fs: Sampling frequency in Hz.
            win_sec: Window duration in seconds for feature extraction.
        """
        self.model_type = model_type
        self.fs = fs
        self.win_sec = win_sec
        
        # These will be set during fit() or load()
        self.raw_mean = None
        self.raw_std = None
        self.scaler = None
        self._is_fitted = False
    
    def _compute_raw_stats(self, X: np.ndarray) -> tuple:
        """Compute per-channel mean and std from data.
        
        Args:
            X: Raw EEG data of shape (N, C, T).
            
        Returns:
            Tuple of (mean, std) arrays of shape (C,).
        """
        mean = X.mean(axis=(0, 2))
        std = X.std(axis=(0, 2)) + 1e-6  # Add epsilon for numerical stability
        return mean, std
    
    def _normalize_raw(self, X: np.ndarray) -> np.ndarray:
        """Normalize raw EEG with stored statistics.
        
        Args:
            X: Raw EEG data of shape (N, C, T).
            
        Returns:
            Normalized data of shape (N, C, T).
        """
        return (X - self.raw_mean[None, :, None]) / self.raw_std[None, :, None]
    
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract DE/PSD features from raw EEG.
        
        Args:
            X: Raw (un-normalized) EEG data of shape (N, C, T).
            
        Returns:
            Features of shape (N, C, F) where F is feature dimension.
        """
        return mlpnet.compute_features(X, fs=self.fs, win_sec=self.win_sec)
    
    def _scale_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features using StandardScaler.
        
        Args:
            features: Feature array of shape (N, C, F).
            fit: If True, fit a new scaler. Otherwise, use existing scaler.
            
        Returns:
            Scaled features of shape (N, C, F).
        """
        orig_shape = features.shape[1:]
        X_2d = features.reshape(len(features), -1)
        
        if fit:
            self.scaler = StandardScaler().fit(X_2d)
        
        X_scaled = self.scaler.transform(X_2d).reshape((len(features),) + orig_shape)
        return X_scaled
    
    def fit(self, X_train: np.ndarray) -> 'EEGPreprocessor':
        """Fit the preprocessor on training data.
        
        This computes normalization statistics and fits the feature scaler
        on the training data only.
        
        Args:
            X_train: Training EEG data of shape (N, C, T).
            
        Returns:
            self for method chaining.
        """
        # Step 1: Compute raw normalization statistics on training data
        self.raw_mean, self.raw_std = self._compute_raw_stats(X_train)
        
        # Step 2: If using glmnet, extract features and fit scaler
        if self.model_type == 'glmnet':
            # IMPORTANT: Extract features from ORIGINAL (un-normalized) data
            features = self._extract_features(X_train)
            self._scale_features(features, fit=True)  # Fit scaler
        
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform EEG data using fitted statistics.
        
        Args:
            X: Raw EEG data of shape (N, C, T).
            
        Returns:
            Processed data of shape (N, C, T+F) for glmnet, or (N, C, T) for others.
            
        Raises:
            RuntimeError: If preprocessor has not been fitted or loaded.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted or loaded before transform.")
        
        # Step 1: Extract features from ORIGINAL data (if needed)
        if self.model_type == 'glmnet':
            features = self._extract_features(X)
            features_scaled = self._scale_features(features, fit=False)
        
        # Step 2: Normalize raw data
        X_normalized = self._normalize_raw(X)
        
        # Step 3: Concatenate if using glmnet
        if self.model_type == 'glmnet':
            return np.concatenate([X_normalized, features_scaled], axis=-1)
        else:
            return X_normalized
    
    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and transform it.
        
        Args:
            X_train: Training EEG data of shape (N, C, T).
            
        Returns:
            Processed training data.
        """
        self.fit(X_train)
        return self.transform(X_train)
    
    def save(self, save_dir: str) -> None:
        """Save preprocessor state to disk.
        
        Args:
            save_dir: Directory to save the state files.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save raw normalization statistics
        np.savez(
            os.path.join(save_dir, 'raw_stats.npz'),
            mean=self.raw_mean,
            std=self.raw_std
        )
        
        # Save scaler if exists
        if self.scaler is not None:
            with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save preprocessor config
        config = {
            'model_type': self.model_type,
            'fs': self.fs,
            'win_sec': self.win_sec,
        }
        with open(os.path.join(save_dir, 'preprocessor_config.pkl'), 'wb') as f:
            pickle.dump(config, f)
    
    def load(self, load_dir: str) -> 'EEGPreprocessor':
        """Load preprocessor state from disk.
        
        Args:
            load_dir: Directory containing the state files.
            
        Returns:
            self for method chaining.
        """
        # Load raw normalization statistics
        stats = np.load(os.path.join(load_dir, 'raw_stats.npz'))
        self.raw_mean = stats['mean']
        self.raw_std = stats['std']
        
        # Load scaler if exists
        scaler_path = os.path.join(load_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load config if exists (for validation)
        config_path = os.path.join(load_dir, 'preprocessor_config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
                # Optionally validate config matches
                if config['model_type'] != self.model_type:
                    print(f"Warning: model_type mismatch. Loaded: {config['model_type']}, Current: {self.model_type}")
        
        self._is_fitted = True
        return self


class EEGDataset(Dataset):
    """PyTorch Dataset for preprocessed EEG data.
    
    This dataset wraps already-preprocessed EEG data for use with DataLoader.
    All preprocessing should be done externally using EEGPreprocessor.
    
    Args:
        data: Preprocessed EEG data of shape (N, C, T+F) or (N, C, T).
        labels: Optional labels of shape (N,).
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray = None):
        """Initialize the dataset.
        
        Args:
            data: Preprocessed EEG data.
            labels: Optional labels array.
        """
        self.data = data.astype(np.float32)
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        # Add channel dimension: (C, T) -> (1, C, T)
        x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x


class EEGInferenceDataset(Dataset):
    """Dataset for inference that applies preprocessing on-the-fly.
    
    This is useful for inference scenarios where we want to process
    data sample-by-sample without loading all data into memory.
    
    Args:
        raw_data: Raw EEG data of shape (N, C, T) or nested structure.
        preprocessor: Fitted EEGPreprocessor instance.
    """
    
    def __init__(self, raw_data: np.ndarray, preprocessor: EEGPreprocessor):
        """Initialize the inference dataset.
        
        Args:
            raw_data: Raw EEG data.
            preprocessor: Fitted preprocessor instance.
        """
        self.raw_data = raw_data.astype(np.float32)
        self.preprocessor = preprocessor
        
        if not preprocessor._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before use in dataset.")
    
    def __len__(self) -> int:
        return len(self.raw_data)
    
    def __getitem__(self, idx: int):
        # Get single sample and add batch dimension for preprocessing
        sample = self.raw_data[idx]  # (C, T)
        sample_batch = sample[np.newaxis, ...]  # (1, C, T)
        
        # Apply preprocessing
        processed = self.preprocessor.transform(sample_batch)  # (1, C, T+F)
        
        # Remove batch dimension and add channel dimension
        x = torch.tensor(processed[0], dtype=torch.float32).unsqueeze(0)  # (1, C, T+F)
        return x


def create_datasets(
    raw_data: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    model_type: str = 'glmnet',
    fs: int = 200,
    win_sec: float = 1.0,
) -> tuple:
    """Create train/val/test datasets with proper preprocessing.
    
    This function encapsulates the complete data preparation pipeline:
    1. Split data into train/val/test
    2. Fit preprocessor on training data only
    3. Apply preprocessing to all splits
    4. Create Dataset objects
    
    Args:
        raw_data: Raw EEG data of shape (N, C, T).
        labels: Labels of shape (N,).
        train_idx: Indices for training set.
        val_idx: Indices for validation set.
        test_idx: Indices for test set.
        model_type: Model architecture type.
        fs: Sampling frequency.
        win_sec: Window duration for feature extraction.
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, preprocessor).
    """
    # Split data
    X_train, y_train = raw_data[train_idx], labels[train_idx]
    X_val, y_val = raw_data[val_idx], labels[val_idx]
    X_test, y_test = raw_data[test_idx], labels[test_idx]
    
    # Create and fit preprocessor on training data
    preprocessor = EEGPreprocessor(model_type=model_type, fs=fs, win_sec=win_sec)
    
    # Transform all splits
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Create datasets
    train_dataset = EEGDataset(X_train_processed, y_train)
    val_dataset = EEGDataset(X_val_processed, y_val)
    test_dataset = EEGDataset(X_test_processed, y_test)
    
    return train_dataset, val_dataset, test_dataset, preprocessor
