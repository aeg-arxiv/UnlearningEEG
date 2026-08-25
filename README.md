# Auditing Machine Unlearning in EEG Biometrics

Code for the paper "Auditing Machine Unlearning in EEG Biometrics: Are
Identities Truly Forgotten?", ICCE Berlin 2026.

Five methods are evaluated for removing an enrolled subject from an EEG
identification model, across four classifiers and two public datasets. Residual
identity is measured through aggregate prediction entropy, prediction
consistency, and a membership inference formulation.

Classifiers: Random Forest, Extra Trees, SGD (modified Huber loss), and Ridge.

Datasets

PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB) — 109 subjects,
64 channels at 160 Hz. Run R03 (motor execution) trains, run R04 (motor
imagery) tests, giving a cross-paradigm protocol. Download from
[PhysioNet](https://physionet.org/content/eegmmidb/)

Lee2019-54 subjects, 62 channels at 1000 Hz, two sessions on separate
days. Session 1 trains, session 2 tests, giving a cross-session protocol.
Retrieved automatically through [MOABB](https://github.com/NeuroTechX/moabb).

Extract features first. Both scripts write to `data/`.

python extract_eegmmidb_nooverlap.py
python extract_lee2019_nooverlap.py

Then run any method. Each writes a summary JSON with per-subject results and
resumes from a partial file if interrupted.

Runtimes vary considerably. Naive deletion trains one model and finishes in
minutes; exact retraining and the perturbation methods retrain per target
subject and can take many hours, particularly on Lee2019 where each subject
contributes roughly 715 windows per session.

Requirements

Python 3.11 with numpy, scipy, scikit-learn, matplotlib, mne, and moabb.

```
pip install numpy scipy scikit-learn matplotlib mne moabb
```

Results were produced with fixed random seeds, so reruns on the same feature
files reproduce the reported values.

## License

MIT. See `LICENSE`.
