import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import argparse

def seg_sliding_window(data, win_s, step_s, fs=200):
    """Segment data into sliding windows.
    data : np.ndarray
        Input data of shape (7, 40, 5, 62, 2 * fs)
    win_s : float
        Window size in seconds (e.g., 0.5 for 500 ms)
    step_s : float
        Step size in seconds (e.g., 0.25 for 250 ms)
    fs : int
        Sampling frequency in Hz (default 200 Hz)"""
    
    win_t = int(fs * win_s)   # number of time points per window (100)
    step_t = int(fs * step_s) # step between windows (50)
    # Sliding window along the time axis (-1)
    windows = sliding_window_view(data, window_shape=win_t, axis=-1)
    # windows.shape -> (7, 40, 5, 62, 301, 100)

    # Subsample with step STEP_T to obtain 7 windows
    windows = windows[..., ::step_t, :]
    # windows.shape -> (7, 40, 5, 62, 7, 100)

    # Rearrange to get (7, 40, 5, 7, 62, 100)
    windows = windows.transpose(0, 1, 2, 4, 3, 5)

    return windows

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', type=str, default='data/Preprocessing/Segmented_Rawf_200Hz_2s')
    p.add_argument('--fs' , type=int, default=200, help='Sampling frequency in Hz')
    # A one-second sliding window with 500 ms overlapping is used for EEG segmentation and frequency feature extraction. shape: (7, 40, 5, 3, 62, 200)
    # if using 0.5s window with 0.25s step, shape: (7, 40, 5, 7, 62, 100)
    p.add_argument('--win_s', type=float, default=1.0, help='Window size in seconds')
    p.add_argument('--step_s', type=float, default=0.5, help='Overlap size in seconds')
    
    return p.parse_args()
    
if __name__ == "__main__":

    # Input directory
    INPUT_DIR = 'data/Preprocessing/Segmented_Rawf_200Hz_2s'

    args = parse_args()

    # Output directory depends on WIN_S
    OUTPUT_DIR = f'data/Preprocessing/Segmented_{int(1000*args.win_s)}ms_sw'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith('.npy'):
            continue

        path_in = os.path.join(INPUT_DIR, fname)
        data = np.load(path_in)  # shape: (7, 40, 5, 62, 400)
        
        # Check if data has the expected shape
        if data.ndim != 5 or data.shape[-1] != 2 * args.fs:
            print(f"Skipping {fname}: unexpected shape {data.shape}")
            continue
        
        windows = seg_sliding_window(data, args.win_s, args.step_s, fs=args.fs)
        
        # Save segmented windows
        out_path = os.path.join(OUTPUT_DIR, fname)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(out_path, windows)

        print(f"Saved segmented windows for {fname} -> {windows.shape}")
        
        
