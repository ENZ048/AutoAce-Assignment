# Backend change requests from the dashboard track

Requests for `src/autoace_audio/` (owned by the backend session on `main`). The dashboard never edits backend code directly.

## 1. Case-insensitive manifest discovery (low priority)

`batch.validate_batch` → `_find_manifest` discovers the manifest with `input_dir.glob("*.csv")`, which is case-sensitive: a batch whose manifest is named `LABELS.CSV` (plausible from Windows/Excel) is treated as "no CSV manifest found" and processed file-discovery-only, with only a warning. Request: match `*.csv` case-insensitively (e.g. `p.suffix.lower() == ".csv"` over `iterdir()`), keeping the existing "first sorted match wins" rule.

Context: found 2026-07-17 while building the dashboard's zip extraction (`src/dashboard/zipsafe.py`), which now relocates root-level CSVs case-insensitively; end-to-end uppercase-manifest support still needs this backend change. Not blocking the dashboard — the validation-report warning surfaces the situation honestly.
