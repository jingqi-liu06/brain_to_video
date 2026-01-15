from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

def standard_scale_features(X, scaler=None, return_scaler=False):
    """Scale features with ``StandardScaler``.

    Parameters
    ----------
    X : np.ndarray
        Array of shape ``(N, ...)`` to scale.
    scaler : sklearn.preprocessing.StandardScaler or None
        If ``None`` a new scaler is fitted on ``X``. Otherwise ``X`` is
        transformed using the provided scaler.
    return_scaler : bool, optional
        Whether to return the fitted scaler.

    Returns
    -------
    np.ndarray
        Scaled array with the same shape as ``X``.
    sklearn.preprocessing.StandardScaler, optional
        Returned only if ``return_scaler`` is ``True``.
    """

    orig_shape = X.shape[1:]
    X_2d = X.reshape(len(X), -1)

    if scaler is None:
        scaler = StandardScaler().fit(X_2d)

    X_scaled = scaler.transform(X_2d).reshape((len(X),) + orig_shape)

    if return_scaler:
        return X_scaled, scaler
    return X_scaled


def compute_raw_stats(X: np.ndarray):
    """Compute per-channel mean and std from training data."""
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2)) + 1e-6
    return mean, std


def normalize_raw(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Normalize raw EEG with provided statistics."""
    return (X - mean[None, :, None]) / std[None, :, None]


def load_scaler(path: str) -> StandardScaler:
    """Load a ``StandardScaler`` object from ``path``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_raw_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load raw EEG normalization statistics from a ``.npz`` file."""
    data = np.load(path)
    return data["mean"], data["std"]


def subject_split(
    seed: int,
    subjects: list[str],
    ckpt_seed_dir: str,
    n_select: int | None = None,
    n_train: int = 13,
    n_val: int = 2,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically split subjects for multi-subject training.

    Parameters
    ----------
    seed : int
        Seed controlling the shuffle.
    subjects : list[str]
        Available subject names.
    ckpt_seed_dir : str
        Directory where ``subjects.txt`` will be written.
    n_select : int, optional
        Number of subjects drawn from ``subjects`` before splitting.
        If ``None`` all subjects are considered.
    n_train : int, optional
        Number of subjects used for training.
    n_val : int, optional
        Number of subjects used for validation.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Lists of train, validation and test subjects.
    """

    rng = np.random.default_rng(seed)
    subj_sorted = sorted(subjects)
    rng.shuffle(subj_sorted)

    if n_select is not None:
        if n_select > len(subj_sorted):
            raise ValueError("Not enough subjects to select")
        selected = subj_sorted[:n_select]
    else:
        selected = subj_sorted

    if n_train + n_val > len(selected):
        raise ValueError("Not enough subjects to split")

    train_subj = selected[:n_train]
    val_subj = selected[n_train : n_train + n_val]
    test_subj = [s for s in subj_sorted if s not in train_subj and s not in val_subj]

    os.makedirs(ckpt_seed_dir, exist_ok=True)
    txt_path = os.path.join(ckpt_seed_dir, "subjects.txt")
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write("train:" + ",".join(train_subj) + "\n")
            f.write("val:" + ",".join(val_subj) + "\n")
            f.write("test:" + ",".join(test_subj) + "\n")

    return train_subj, val_subj, test_subj


def block_split(seed: int, n_blocks: int, ckpt_seed_dir: str) -> tuple[int, int]:
    """Choose validation and test blocks for single-subject training.

    Parameters
    ----------
    seed : int
        Seed controlling the selection.
    n_blocks : int
        Total number of blocks in the data.
    ckpt_seed_dir : str
        Directory where ``blocks.txt`` will be written.

    Returns
    -------
    tuple[int, int]
        Selected validation and test block indices.
    """

    rng = np.random.RandomState(seed)
    val_block, test_block = rng.choice(np.arange(n_blocks), size=2, replace=False)

    os.makedirs(ckpt_seed_dir, exist_ok=True)
    txt_path = os.path.join(ckpt_seed_dir, "blocks.txt")
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write(f"val:{int(val_block)}\n")
            f.write(f"test:{int(test_block)}\n")

    return int(val_block), int(test_block)
