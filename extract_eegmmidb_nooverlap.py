import os
import mne
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch


def extract_band_power(window, sfreq=160):
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "mu": (8, 12),
        "beta": (13, 30),
    }

    band_feats = []

    for ch in range(window.shape[0]):
        freqs, psd = welch(
            window[ch],
            fs=sfreq,
            nperseg=min(256, window.shape[1])
        )

        for low, high in bands.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_feats.append(np.mean(psd[idx]))

    return np.array(band_feats)


def extract_features(data, sfreq=160):
    mean_feat = np.mean(data, axis=1)
    std_feat = np.std(data, axis=1)
    median_feat = np.median(data, axis=1)
    range_feat = np.ptp(data, axis=1)
    skew_feat = skew(data, axis=1)
    kurt_feat = kurtosis(data, axis=1)

    band_feat = extract_band_power(data, sfreq=sfreq)

    return np.concatenate([
        mean_feat,
        std_feat,
        median_feat,
        range_feat,
        skew_feat,
        kurt_feat,
        band_feat
    ])


base_dir = "physionet-eegmmidb/files"

window_size = 320
step_size = 320
sfreq = 160

runs_to_use = ["R03", "R04"]

features = []
labels = []
groups = []

subjects = sorted([
    d for d in os.listdir(base_dir)
    if d.startswith("S")
])

for subj in subjects:
    for run in runs_to_use:
        file_path = os.path.join(base_dir, subj, f"{subj}{run}.edf")

        if not os.path.exists(file_path):
            print(f"Missing: {subj} {run}")
            continue

        try:
            group_id = f"{subj}_{run}"

            raw = mne.io.read_raw_edf(
                file_path,
                preload=True,
                verbose=False
            )

            raw.filter(1.0, 40.0, verbose=False)
            raw.set_eeg_reference(
                "average",
                projection=True,
                verbose=False
            )
            raw.apply_proj(verbose=False)

            data = raw.get_data()
            n_samples = data.shape[1]

            count = 0

            for start in range(
                0,
                n_samples - window_size + 1,
                step_size
            ):
                window = data[:, start:start + window_size]
                feat_vector = extract_features(window, sfreq=sfreq)

                features.append(feat_vector)
                labels.append(subj)
                groups.append(group_id)
                count += 1

            print(f"{subj} {run}: {count} non-overlapping windows")

        except Exception as e:
            print(f"Error {subj} {run}: {e}")


features = np.array(features)
labels = np.array(labels)
groups = np.array(groups)

np.save("features_spectral_nonoverlap.npy", features)
np.save("labels_spectral_nonoverlap.npy", labels)
np.save("groups_spectral_nonoverlap.npy", groups)


print(f"Features shape : {features.shape}")
print(f"Labels shape   : {labels.shape}")
print(f"Groups shape   : {groups.shape}")
print(f"Unique subjects: {len(set(labels))}")
print(f"Unique groups  : {len(set(groups))}")
