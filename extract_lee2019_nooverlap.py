"""
LEE2019 FEATURE EXTRACTION
"""

from pathlib import Path
import numpy as np
import mne
from scipy.stats import skew, kurtosis
from moabb.datasets import Lee2019_MI
import warnings
warnings.filterwarnings("ignore")

OUTPUT_PATH = Path("data")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Parameters

SFREQ       = 1000
WINDOW_SIZE = 2000          
STEP_SIZE   = 2000          
N_FFT       = 1024
FMIN        = 1.0
FMAX        = 40.0

BANDS = {
    "delta":     (1,  4),
    "theta":     (4,  8),
    "alpha":     (8,  13),
    "beta":      (13, 30),
    "low_gamma": (30, 40),
}

SESSION_MAP = {
    "0": "R03",
    "1": "R04",
}

SUBJECTS = list(range(1, 55))

print("=" * 70)
print("LEE2019 FEATURE EXTRACTION — NON-OVERLAPPING")
print("=" * 70)
print(f"Subjects: {len(SUBJECTS)}")
print(f"Window: {WINDOW_SIZE} samples ({WINDOW_SIZE/SFREQ:.1f}s) at {SFREQ} Hz")
print(f"Step:   {STEP_SIZE} samples (NON-OVERLAPPING)")

dataset = Lee2019_MI()

X_stat  = []
X_psd   = []
y_all   = []
runs_all = []

for subj in SUBJECTS:
    try:
        raw_data = dataset.get_data(subjects=[subj])
    except Exception as e:
        print(f"\nSubject {subj}: ERROR — {e}")
        continue

    subject_id = f"S{subj:03d}"

    for session_key, run_label in SESSION_MAP.items():
        if session_key not in raw_data[subj]:
            continue

        session_data = raw_data[subj][session_key]

        for run_key, run_data in session_data.items():
            raw = run_data
            raw.load_data(verbose=False)
            raw.pick_types(eeg=True, emg=False, stim=False, verbose=False)
            raw.filter(FMIN, FMAX, fir_design="firwin", verbose=False)
            raw.set_eeg_reference("average", projection=False, verbose=False)

            data = raw.get_data()
            n_channels, n_samples = data.shape

            count = 0
            for start in range(0, n_samples - WINDOW_SIZE + 1, STEP_SIZE):
                window = data[:, start:start + WINDOW_SIZE]

                # statistical features
                feat_stat = np.concatenate([
                    np.mean(window, axis=1),
                    np.std(window, axis=1),
                    np.median(window, axis=1),
                    np.ptp(window, axis=1),
                    skew(window, axis=1),
                    kurtosis(window, axis=1),
                ])

                # PSD features
                psds, freqs = mne.time_frequency.psd_array_welch(
                    window, sfreq=SFREQ, fmin=FMIN, fmax=FMAX,
                    n_fft=N_FFT, verbose=False)

                band_powers = []
                for _, (low, high) in BANDS.items():
                    mask = (freqs >= low) & (freqs < high)
                    bp = psds[:, mask].mean(axis=1)
                    band_powers.extend(bp)

                feat_psd_scaled = np.log1p(np.array(band_powers) * 1e9)

                X_stat.append(feat_stat)
                X_psd.append(feat_psd_scaled)
                y_all.append(subject_id)
                runs_all.append(run_label)
                count += 1

    print(f"S{subj:03d}: done")

X_stat   = np.array(X_stat)
X_psd    = np.array(X_psd)
y_all    = np.array(y_all)
runs_all = np.array(runs_all)

print(f"X_statistical shape:    {X_stat.shape}")
print(f"X_scaled_log_psd shape: {X_psd.shape}")
print(f"y shape:                {y_all.shape}")
print(f"runs shape:             {runs_all.shape}")
print(f"Unique subjects:        {len(np.unique(y_all))}")
print(f"R03 windows:            {(runs_all == 'R03').sum()}")
print(f"R04 windows:            {(runs_all == 'R04').sum()}")

np.save(OUTPUT_PATH / "X_lee2019_scaled_log_psd_nooverlap.npy", X_psd)
np.save(OUTPUT_PATH / "X_lee2019_statistical_nooverlap.npy",    X_stat)
np.save(OUTPUT_PATH / "y_lee2019_nooverlap.npy",                y_all)
np.save(OUTPUT_PATH / "runs_lee2019_nooverlap.npy",             runs_all)

