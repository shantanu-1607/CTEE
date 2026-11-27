# Prediction example

`recordings.csv` contains four complete recordings selected from development
houses, one for each supplied load label. It keeps only the ID, time/sample
columns and required electrical measurements. It has no labels, appliance names
or house identifiers.

Run from the repository root:

```sh
python -m load_classifier predict --input examples/recordings.csv --output /tmp/load-predictions.csv
```

The destination must be a new file. These recordings demonstrate the input
format and command; they are not an independent evaluation set.
