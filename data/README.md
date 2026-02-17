# Data (manifests only)

This repository does not include raw audio files.

It only provides CSV manifests that describe the dataset splits used in the experiments.

## Folder structure

- `data/manifests/`: CSV manifests for Train/Val/Test splits.

## Notes on labels (BirdNET)

Some manifests and/or splits include a `*_BirdNET.csv` version.

These are equivalent to the original splits, but the class (folder) names are adapted to match BirdNET’s label naming convention, ensuring a direct mapping between folder names and BirdNET output labels.

## Files

- `dataset_manifest_TRAIN_*.csv`: training splits (various training-set sizes)
- `dataset_manifest_VAL*.csv`: validation split(s)
- `dataset_manifest_TEST_9.csv`: test split restricted to the 9 target species
- `dataset_manifest_TEST_40.csv`: test split for the 40-species setting
- `dataset_manifest_*_BirdNET.csv`: same split but with BirdNET-compatible label/folder naming