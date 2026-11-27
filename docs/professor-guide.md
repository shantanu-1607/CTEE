# Explaining the project

## A short explanation to say aloud

“We built a supervised machine-learning model that predicts one of four
provided load labels: R, I, RC or RLC. We matched 1,074 JSON entries to their
electrical recordings. Each recording contains measurements over time.
We summarized the electrical behaviour and its changes over time, compared
several classifiers, and selected Extra Trees using validation data from
separate houses. On 215 test recordings from 12 houses the model had never
trained on, it predicted 202 correctly: 93.95% accuracy and 0.9275 macro-F1.
The main limitation is confusion between I and RLC.”

## What changed from the previous project?

The previous workbook was a simulated dataset with two answers: CC/CP/CZ
behaviour and one of seven circuit labels. The replacement data has a different
schema and only four supplied load labels. We therefore train one four-class
classifier. We cannot claim CC/CP/CZ prediction because no such ground-truth
labels are present. We preserve the name I exactly as supplied rather than
silently interpreting or renaming it.

The old 98%/97% results belong to another dataset and another task. Comparing
them directly with the new 93.95% would be misleading.

## What exactly is in the data?

- The JSON contains 1,074 recording descriptions, labels and house identifiers.
- The matching combined CSV contains 209,554 electrical-feature windows.
- A window represents 500 samples measured at 30,000 samples per second.
  That is 500/30,000 = 1/60 second, approximately 16.67 milliseconds.
- The recordings cover 55 houses and 11 appliance types.
- Labels by recording: R = 305, I = 191, RC = 347, RLC = 231.

The CSV contains precomputed quantities such as RMS current and voltage, power,
power factor, phase information, waveform shape and harmonics. We did not extract
those quantities from raw waveforms ourselves. The raw waveforms and original
feature-extraction code were not part of the supplied Drive files.

Every JSON recording ID matched the CSV. Labels, houses and appliance types
agreed for every recording. We checked duplicate window keys, timestamps,
sample indices, missing values and infinite values. Source file hashes record
exactly which files produced the model.

## How do we use timestamps?

We sort a recording's windows by elapsed time. For each of 23 measurements,
we calculate:

1. A low value: the 10th percentile.
2. A typical value: the median.
3. A high value: the 90th percentile.
4. Spread: the middle 50% range, called the interquartile range.
5. The typical value during the first 20% of elapsed time.
6. The typical value during the last 20% of elapsed time.
7. Its overall upward or downward slope per second.
8. Its typical absolute change per second between adjacent windows.

That is 23 × 8 = 184 summaries. These describe both what the signal looks like
and how it changes. The imputer also adds 64 indicators showing which summaries
were missing. The final forest receives 248 numbers.

For example, two recordings might have similar average current, but one starts
with a higher current that falls while the other stays steady. Early/late values
and slopes let the model distinguish those patterns. This example explains
what the features can represent; it does not prove which specific feature
caused an individual prediction.

The date “July, 2013” is identical for all entries. It is not a useful predictor.
We use time within recordings, not time of day or calendar forecasting.

## Why one prediction per recording?

The load label describes a recording, not an independently labelled electrical
cycle. Adjacent windows are closely related. Treating 209,554 windows as
209,554 independent labelled appliances would exaggerate how much independent
data we have. We summarize each recording and evaluate 1,074 recording-level
examples instead. The model needs the complete recording and is currently
an offline classifier, not a live streaming detector.

## Why Extra Trees rather than a neural network?

We compared them under the same development protocol. For the tested settings:

| Model | Mean validation accuracy | Mean validation macro-F1 |
| --- | ---: | ---: |
| Selected Extra Trees | 92.39% | 0.9056 |
| Compact Extra Trees | 91.36% | 0.8932 |
| Gradient boosting | 89.22% | 0.8669 |
| Random forest | 91.13% | 0.8911 |
| Neural network | 82.87% | 0.8007 |
| Logistic regression | 82.90% | 0.8050 |
| SVM, C = 10 | 80.78% | 0.7809 |

The full comparison has 12 candidate settings, including a majority baseline.
Three Extra Trees variants were added after the first development comparison.
All choices used development results; the test set was still reserved.

Extra Trees combines many decision trees. A tree asks numerical questions such
as whether a signal summary is above or below a threshold. Extra Trees tries
random thresholds and chooses useful splits. The selected setting uses 400
trees, considers half of the input columns at each split and combines their class
probabilities. Class weights give less common labels more weight during fitting.

This worked better for our precomputed tabular features and small number of
labelled recordings. It does not prove that every neural network is worse.
A better sequence model or a larger independently labelled dataset could
change the comparison.

## How did we split and train fairly?

We reserved 215 recordings from 12 houses for testing. The remaining 859
recordings from 43 houses formed the development set.

Within development, four-fold cross-validation rotates which houses provide
validation examples. A model trains on three folds and predicts the fourth.
After all four rotations, each development recording has a prediction from
a model that did not train on its house. We select the classifier with the
highest mean fold macro-F1, then train that setting on all development recordings.
Finally, we evaluate it on the reserved test houses.

We use fixed random seeds and save the exact split. No house crosses the
development/test boundary or appears in multiple validation folds. The imputer
learns missing-value replacements only from the current training data. Scaling,
where needed by linear models, SVMs and the neural network, follows the same rule.
Extra Trees does not require feature scaling.

The deeper duplicate check found five pairs with identical electrical/time
input windows across house11 and house19. Both houses belong to development.
We link those houses into one group so their recordings cannot appear on
opposite sides of a validation split. This leaves 42 connected groups across
the 43 development houses. The 12 reserved test houses are unchanged and have
no duplicate summary matching development. We retain all supplied IDs, while
acknowledging that only 1,069 summary rows are distinct.

This duplicate connection was found after an initial test evaluation. We
corrected the validation folds and repeated the same 12 candidate settings.
The corrected validation scores selected the final model. Test scores were
not used for that selection. We report the final re-evaluation and do not
claim the test set had never previously been inspected.

We exclude labels, appliance names, house identifiers, recording IDs, collection
dates and recording length as direct model inputs. These fields are retained
only for matching, splitting and audits. The feature allowlist also prevents
accidental use of extra CSV columns at inference.

House grouping addresses a major source of overlap. It cannot certify physical
appliance independence across houses because the supplied metadata has no
reliable physical appliance identifier. We also cannot independently audit the
original feature extractor for how its input quantities were computed.

## What do the final numbers mean?

Test accuracy is 202/215 = **93.95%**. It answers: “What fraction of complete
test recordings received the correct supplied load label?”

The majority baseline always predicts RC, the most common development label.
It gets 68/215 = **31.63%** on test. That makes the learned result meaningfully
better than simply guessing the most frequent label.

Macro-F1 is **0.9275**. It calculates F1 for each of the four classes and gives
each class equal weight, so a smaller class is not hidden by the larger ones.
Balanced accuracy is **92.50%**, the mean recall across the four classes.

| Class | Precision | Recall | F1 | Test support |
| --- | ---: | ---: | ---: | ---: |
| R | 100.00% | 95.00% | 0.9744 | 60 |
| I | 100.00% | 75.00% | 0.8571 | 40 |
| RC | 100.00% | 100.00% | 1.0000 | 68 |
| RLC | 78.33% | 100.00% | 0.8785 | 47 |

Precision means: “When the model predicts this class, how often is it right?”
Recall means: “Of the actual examples of this class, how many did it find?”
F1 balances precision and recall. Support is the number of actual examples.

For RLC, recall is 100% because all 47 actual RLC recordings were found. Its
precision is only 78.33% because 13 other recordings were also called RLC.
For I, all 30 I predictions were correct, but 10 of the 40 actual I recordings
were missed. That gives 100% precision and only 75% recall.

## Which mistakes were made?

Ten I recordings and three R recordings were predicted as RLC. The I errors
were seven Fan recordings and three Fridge recordings. The R errors were three
Heater recordings. These errors are saved individually with their IDs and houses.
There were no other error directions on this particular test split.

Possible explanations include overlapping electrical signatures, insufficient
representative examples, measurement quality and coarse load annotations.
The confusion matrix establishes the errors; it does not establish their
physical cause. Diagnosing the cause needs waveform inspection and label review.

## Is 93% guaranteed on new data? Is the model overfitting?

No score is guaranteed. Training accuracy is 100%, while validation and test
accuracy are about 93%. The forest can fit training examples very closely.
The training number alone therefore tells us little about new recordings.
The similar validation and held-out test results are encouraging evidence for
this collection, rather than proof that overfitting is absent.

Resampling whole test houses gives an approximate 95% accuracy interval of
**89.30%–97.77%**. Whole-house resampling keeps related recordings together.
This interval still depends on the chosen test houses and does not include
uncertainty about the labels or every model-selection decision.

All four load labels occur in test, but only ten of the eleven appliance types
do. Washing Machine has zero test recordings, so we cannot claim a measured
washing-machine-specific test accuracy.

## Does this prove the physical circuit type?

It predicts the supplied annotations. Each appliance type was assigned one load
label in this JSON. Appliance names are excluded from training, but the labels
still come from that annotation scheme. We have no independent equivalent-circuit
measurements proving the assignments. The source also warns that some recordings
are not well calibrated. We should describe the task as predicting the provided
load categories, with this limitation stated openly.

## Did timestamps improve the score?

For the selected Extra Trees setting, median-only summaries scored 0.8899
validation macro-F1 and the full summaries scored 0.9056. That supports using
the fuller representation for this setting. However, the comparison adds
distribution summaries as well as time summaries, so it does not isolate the
benefit of timestamps alone. More features do not automatically mean better accuracy.

## How can we improve it next?

First review I/RLC labels and collect more independent Fan, Fridge and mixed-load
examples across houses. Check calibration and get the original extraction code
and raw signals. Then compare more focused temporal features, other time scales,
feature removal and additional classifier settings using development folds only.
More independent recordings are likely more valuable than simply making a much
larger network from the same 1,074 labels.

A sequence neural network becomes a reasonable experiment if the raw waveforms
or reliable ordered sequences are available and there is enough independent
data. Test it under the same house grouping. For later development, keep a new
external test collection untouched so progress is not judged by repeatedly
optimizing against already inspected test errors.

## Where is each step in the code?

- `data.py`: validates recordings, matches JSON labels and makes house splits.
- `features.py`: turns each ordered recording into 184 summaries.
- `models.py`: defines classifier settings and preprocessing pipelines.
- `experiment.py`: prepares data, runs development comparisons and fits the winner.
- `evaluate.py`: exports reserved-test scores, plots, errors and verification.
- `predict.py`: loads the saved pipeline and predicts new feature CSVs.
- `__main__.py`: connects those steps to command-line commands.

The trained artifact is `runs/timestamped/model.joblib`. Prediction needs only
the required electrical and time columns, not appliance names or load labels.
