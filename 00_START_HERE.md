# CTEE project: start here

**Status: implemented and tested, 22 October 2026.** The original plan/PDF remains in this folder as proposal history. The current implementation uses manual NumPy algorithms and a local dashboard.

The computer reads voltage/current and gives two answers: **CC/CP/CZ behaviour** and **R/L/C branch composition**. Our proposed enhancement adds electrical evidence, confidence checks and extra simulated measurements when needed.

- Start the app: double-click `implementation/Run_Dashboard.command`; open <http://127.0.0.1:8501>.
- Read setup and completed phases: [implementation/README.md](implementation/README.md).
- Read current measured results and limitations: [implementation/RESULTS_NUMPY.md](implementation/RESULTS_NUMPY.md).
- Understand/present it simply: [implementation/docs/VIVA_GUIDE.md](implementation/docs/VIVA_GUIDE.md).
- Inspect the adjustable experiments: [implementation/configs/experiments.json](implementation/configs/experiments.json).
- Check exact class definitions: [implementation/docs/LABEL_CONTRACT.md](implementation/docs/LABEL_CONTRACT.md).

The current held-out simulator result is **100% clean and 96.67% at 20 dB joint accuracy**, versus **93.33% at 20 dB** for the matching noise-trained baseline. Both answers must be right. A separate pool of 140 new circuits gives 99.17% at 20 dB. All 64 tests pass in a clean environment without ML frameworks. Severe coloured noise remains a failure condition; its rejection is reported separately from accuracy.

The source includes manual shared-layer backpropagation, Adam/SGD, Fourier/FFT analysis, circuit fits, transient integration, calibration, metrics and paired statistical comparisons. Plotting/UI libraries are used for display and files. NumPy is allowed by the professor's clarified restriction.

The benchmark is an **ideal controlled-source simulator with a known series branch and 18 supported pairs**. It does not establish real-appliance, arbitrary-topology or hardware accuracy. Confirm this scope with the professor before presenting it as the final interpretation of the assignment. It is possible to demonstrate a useful project contribution without claiming worldwide novelty or a guaranteed grade.

Generated datasets, checkpoints, plots and detailed reports are stored locally in `implementation/runs/numpy_v1/` and ignored by Git. The app loads that new run. Earlier prototype files remain historical; their metrics must not be attributed to the current NumPy implementation.

Local commits use the repository owner's configured identity, without assistant attribution. Nothing is automatically pushed to GitHub.
