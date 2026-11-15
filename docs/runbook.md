# Installation, rebuild and verification

Use 64-bit Python 3.11 on Windows for the tested CPU setup. A virtual environment created without `--system-site-packages` is required for an independent dependency check.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-resolved.txt
python -m pip check
python -m pytest -q
python scripts/prepare_data.py --download
python scripts/train_models.py
python scripts/verify_project.py
python scripts/build_report.py
python scripts/export_presentation_data.py
streamlit run app.py
```

`requirements-lock.txt` records direct versions. The resolved environment file, when supplied, records the dependencies of the independently validated environment. Prefer that resolved file for an exact environment. `requirements.txt` describes supported ranges; it is not an exact lock.

The source archive deliberately excludes MovieLens files and checkpoints. Preprocessing downloads the official archive with TLS verification, writes the SHA256, and extracts it locally. If network access fails, download the official archive separately and place the extracted `.dat` files under `data/raw/ml-1m/`; then run preprocessing without `--download`. Do not disable certificate verification or redistribute the archive.

Preprocessing writes the three processed tables, the inspection summary and comparison table. Training writes model checkpoints, validation trials and the selected test rows. Full training uses the documented CPU prefix caps; epochs are fixed and no early stopping is applied.

## Fast implementation smoke run

To avoid replacing the full result tables, always redirect every generated training output:

```powershell
python scripts/train_models.py --quick --artifacts-dir .smoke/artifacts --results-dir .smoke/results --figures-dir .smoke/figures
```

This uses the default processed MovieLens tables but trains fewer epochs/configurations. Do not publish quick metrics or build the technical report from them. Remove `.smoke` after inspection; it is ignored by Git.

## Re-export and inspect

If saved models already exist, `python scripts/export_unseen_examples.py` refreshes product examples without retraining. It uses exactly the same user selection, unseen filter and score-tie rule as training and the application. Then run `verify_project.py` and rebuild the report/summary.

Verification recomputes all three models' held-out metrics, checks validation selection and chronological bounds, reconstructs every exported example from saved models, and checks full-history exclusion. `results/verification.json` contains the evidence. This verifies scoring consistency; it does not independently rerun optimization from initialization.

## Operational notes

- Training time and test-scoring time are separate in `results/metrics.csv`; selected training durations are also generated into `results/training_info.json`.
- A missing or incompatible old checkpoint requires retraining. Do not mix the old ICSRec-inspired weights with the current architecture.
- Ranking scores are raw utilities; scores from different models do not share a common calibrated scale.
- Apply a filter with no matching unseen movies to check the UI warning; the comparison tabs should remain usable.
- Test results and environment versions describe the recorded run. New package/platform versions may produce small numeric differences; compare the stated tolerance and inspect any ranking changes.
