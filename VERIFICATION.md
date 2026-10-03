# Verification record

## Completed

- Existing `tests/test_data.py`: **2 passed**.
- Python source compilation: **passed**.
- Validation notebook JSON parsing: **passed**.
- Repository-wide privacy/path scan: **passed**; no `/Users/`, `/home/`, `/kaggle/`, personal username, Downloads/Desktop/OneDrive, or `/private/tmp` paths remain.
- Supplied `SP500_master_panel.csv`: **349,440 rows × 31 columns** and contains all columns required by the AHRN data loader.
- Raw Bloomberg-derived data removed from the final repository package.

## Not completed in this runtime

The AHRN experiment itself could not be executed because the runtime does not have `hmmlearn`, which is imported by `ahrn/models.py`. Installing it was attempted, but this runtime has no external package-network access. The experiment was therefore **not falsely reported as executed**.

Run after installing requirements in a network-enabled environment:

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/check_data.py
python run_experiment.py --config configs/smoke.yaml
python run_experiment.py --config configs/default.yaml
```

The supplied Bloomberg extraction materials do not contain the original Bloomberg field identifiers or terminal/Excel formulas, so those were not fabricated in the extraction guide.
