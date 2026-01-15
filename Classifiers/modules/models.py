"""
Different EEG encoders for comparison

deepnet, eegnet, shallownet, mlpnet, glmnet
"""

import math
import numpy as np
import sys
sys.path.append("/baai-cwm-vepfs/cwm/jingqi.liu/brain_video/codebase/EEG2Video/")
from EEG_preprocessing.DE_PSD import DE_PSD

import torch
import torch.nn as nn

class deepnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(deepnet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (C, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )

        # compute output dimension using a dummy input tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            out_features = self.net(dummy).view(1, -1).shape[1]
        self.out = nn.Linear(out_features, out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
    
class eegnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(eegnet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )

        # compute output dimension using a dummy input tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            out_features = self.net(dummy).view(1, -1).shape[1]
        self.out = nn.Linear(out_features, out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class shallownet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(shallownet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            #nn.AdaptiveAvgPool2d((1, 26)),
            nn.Dropout(0.5),
        )
        n_samples = math.floor((T - 75) / 5 + 1)
        self.out = nn.Linear(40 * n_samples, out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim):
        super(mlpnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )

    @staticmethod
    def compute_features(raw: np.ndarray, fs: int = 200, win_sec: float = 0.5) -> np.ndarray:
        """Compute DE features from raw EEG."""
        feats = np.zeros((raw.shape[0], raw.shape[1], 5), dtype=np.float32)
        for i, seg in enumerate(raw):
            de = DE_PSD(seg, fs, win_sec, which="de")
            feats[i] = de
        return feats
        
    def forward(self, x):               #input:(batch,C,5)
        out = self.net(x)
        return out
    
class glmnet(nn.Module):
    """ShallowNet (raw) + MLP (freq) → concat → FC."""

    def __init__(self, occipital_idx, C: int, T: int, *, feat_dim: int = 5, out_dim: int = 40, emb_dim: int = 512):
        """Construct the GLMNet model.

        Parameters
        ----------
        occipital_idx : iterable
            Indexes of occipital channels used for the local branch.
        C : int
            Number of EEG channels.
        T : int
            Temporal length of the raw EEG windows.
        feat_dim : int, optional
            Number of features per channel in the spectral representation.
            Defaults to ``5``.
        out_dim : int
            Dimension of the classification output.
        emb_dim : int
            Dimension of the intermediate embeddings (each branch outputs
            ``emb_dim`` features).
        """
        super().__init__()
        self.occipital_idx = list(occipital_idx)
        self.time_len = T

        # Global branch processing raw EEG
        self.raw_global = shallownet(emb_dim, C, T)
        # Local branch processing spectral features
        self.freq_local = mlpnet(emb_dim, len(self.occipital_idx) * feat_dim)

        # Projection of concatenated features followed by classifier
        self.projection = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim // 2),
        )
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim // 2, out_dim),
        )

    @staticmethod
    def infer_out_dim(state: dict) -> int:
        """Infer ``out_dim`` from a checkpoint state dict."""
        if "classifier.1.weight" in state:
            return state["classifier.1.weight"].shape[0]
        if "fc.2.weight" in state:
            return state["fc.2.weight"].shape[0]
        if "fc.weight" in state:
            return state["fc.weight"].shape[0]
        raise KeyError("Cannot infer output dimension from checkpoint")

    @staticmethod
    def infer_feat_dim(state: dict, occipital_len: int) -> int:
        """Infer ``feat_dim`` from a checkpoint state dict."""
        for key in (
            "freq_local.net.1.weight",
            "module.freq_local.net.1.weight",
        ):
            if key in state:
                in_features = state[key].shape[1]
                return in_features // occipital_len
        raise KeyError("Cannot infer feature dimension from checkpoint")

    @staticmethod
    def infer_time_len(state: dict) -> int:
        """Infer ``T`` (window length) from a checkpoint state dict."""
        for key in (
            "raw_global.out.weight",
            "module.raw_global.out.weight",
        ):
            if key in state:
                in_features = state[key].shape[1]
                n_samples = in_features // 40
                return (n_samples - 1) * 5 + 75
        raise KeyError("Cannot infer time length from checkpoint")

    def forward(self, x, return_features: bool = False):
        """Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated raw EEG and features of shape
            ``(B, C, time_len + feat_dim)``.
        return_features : bool, optional
            If ``True`` returns the projected features before the
            classification layer. Defaults to ``False``.
        """

        x_raw = x[..., :self.time_len]
        x_feat = x[..., self.time_len:].squeeze(1)

        g_raw = self.raw_global(x_raw)
        l_freq = self.freq_local(x_feat[:, self.occipital_idx, :])

        features = torch.cat([g_raw, l_freq], dim=1)
        projected = self.projection(features)

        if return_features:
            return projected

        return self.classifier(projected)
