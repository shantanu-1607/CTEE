# Dataset

`load_dataset_baseline_v3.xlsx` is the supplied final workbook, copied without
changes. The Dataset sheet contains 16,800 examples, 22 measurement columns and
two labels. Every behaviour/circuit pair has 800 examples: 400 normal and 400 tough.

The workbook documents simulated labelled loads whose sizes and supply conditions
use IIT Kanpur and All-India load-profile values. Its behaviour labels describe
CC-, CP- or CZ-dominated ZIP loads. These are not independently measured appliance
labels. The generator and underlying source files are not included here, so that
description is attributed to the workbook rather than independently verified.

The workbook cites Singh, S., *All India Load Profile Data and IIT Kanpur Load
Profile Data*, Mendeley Data V1 (2023), DOI
[10.17632/dyd858nvmc.1](https://doi.org/10.17632/dyd858nvmc.1), CC BY 4.0.
Its notes state that the source units were assumed to be kW during generation.

`splits.csv` assigns every sample to training, validation or test. Allocation is
70/15/15 within each behaviour/circuit/condition group, with seed 42. Do not change
this file to improve an experiment's score. `audit.json` records the workbook
hash, feature list, checks and exact partition counts.

Three columns exactly repeat measurements at normal voltage: V_L2 = Vrms,
I_L2 = Irms and P_L2 = P. The baseline drops those copies and uses the remaining
19 columns. Labels, sample IDs and Source link metadata never enter the network.
Source link is used only for condition reporting and provenance checks.
