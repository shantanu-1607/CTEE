# Supplied timestamped dataset

`meta_labeled.json` is the exact supplied JSON: 1,074 recording IDs, house
identifiers, appliance types and load labels R (305), I (191), RC (347) and
RLC (231). These are recording counts, not independent appliance counts.
No CC/CP/CZ target is present.

The electrical data came from the user's [Drive folder](https://drive.google.com/drive/folders/1_aO3gFYwigITWH0YmCPAoB48wRg2B5mF).
The combined `all_features.csv` contains 209,554 rows and 64 columns and is
214,166,054 bytes. Its exact SHA-256 is stored in `source.json` and the audit.
Individual files `1073_features.csv` and `1074_features.csv` were inspected
before downloading the combined file. Every combined-file ID, load label,
house and appliance type matches the JSON. All 1,074 recordings are present;
no duplicate recording/window keys or infinite measurement values were found.

Each row is a precomputed electrical-feature window. Timestamps and indices
agree with 500 samples per window at 30,000 Hz. `end_sample` is inclusive;
`t_end_s` is exclusive. Windows are checked for order and overlap. Recordings
contain 60–806 windows. JSON's rounded recording lengths agree with final
timestamps to within 0.00334 seconds.

Missing harmonic values occur in 0.94% of original windows. Equivalent L/C
fields have more missing values and are excluded entirely. Missing recording
summaries stay missing until the training-fold imputer fills them. Validation
and test values never fit imputation or scaling.

`processed/recordings.csv` contains one row per recording with audit metadata
and 184 allowed model features. `processed/splits.csv` assigns development/test
partitions and development validation folds. `processed/audit.json` records
counts, missing values, feature names and hashes. Prediction reads only the
allowed feature names, never metadata columns left in these tables.

The schema, collection date, 30 kHz rate and recording count are consistent
with original PLAID as described by the [authors](https://github.com/jingkungao/PLAID).
This is a provenance inference, not verification that the supplied feature
extractor or added load labels belonged to that release. No independently
downloaded PLAID waveform version is substituted. The authoritative inputs
are the user's Drive feature file and labelled JSON.

The JSON warns that some instances are not well calibrated and metadata is
incomplete. Raw waveforms, extraction code, independent circuit measurements,
physical appliance IDs and the label-assignment process were not supplied.
We validate consistency and predict the supplied annotations without asserting
that those annotations prove physical circuit topology.
