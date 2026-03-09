# Copilot instructions for RF Impairments and Modulation Quality Analyzer

## Big picture
- This repo is a single-page Streamlit app. `rf_analyzer/main.py` is only the launcher; the real application orchestration lives in `rf_analyzer/gui/main_window.py`.
- `run_app()` builds the full signal-processing pipeline on every Streamlit rerun: source bits -> modulation -> impairments -> channel noise -> ADC quantization -> metrics -> rendering.
- The core app state is a `controls` dict sourced from `st.session_state` via `rf_analyzer/gui/controls_panel.py`. New features should usually start by adding a default in `DEFAULT_CONTROLS`, exposing a widget in `render_controls()`, and then consuming the value in `run_app()`.
- UI rendering is split out, but not the domain logic: `rf_analyzer/gui/plots_panel.py` expects a `stage_results` list of dicts with keys like `name`, `symbols`, `ideal_symbols`, `freqs`, `psd_db`, and nested `metrics`.

## Current module boundaries
- Treat `rf_analyzer/gui/main_window.py` as the source of truth for modulation, demodulation, impairment application, and metric calculations. Many files under `rf_analyzer/metrics/`, `rf_analyzer/impairments/`, and `rf_analyzer/signal/` are empty or only partial prototypes.
- The one helper that is actually wired into the main flow is `rf_analyzer/impairments/adc_quantization.py`, imported by `main_window.py` through `_quantize_complex_signal()`.
- `rf_analyzer/utils/fft_utils.py` provides the power spectrum used by the per-stage metric calculation.
- `Top-Level Window Structure.txt` documents the intended four-region layout and matches the fixed Streamlit layout implemented in `plots_panel.py`.

## Project-specific patterns
- `stage_results` always contains exactly four stages in this order: `IQ MODULATOR`, `IMPARMENTS`, `CHANNEL`, `ADC`. `plots_panel.py` hardcodes four display columns around that assumption.
- `render_visualizations()` and `render_metrics()` expect each stage dict to already contain fully computed arrays and formatted metric values are derived from `stage["metrics"]`; do computation before rendering.
- The code mixes tabs and spaces across files. Preserve the existing indentation style per file instead of reformatting unrelated lines.
- Some naming is intentionally inconsistent with ideal spelling (`IMPARMENTS`, `pa_1dbcp_dbm`, `random data source.py`). Keep public keys, labels, and filenames stable unless the change is explicitly about renaming.
- Streamlit styling is done inline with `st.markdown(..., unsafe_allow_html=True)` blocks inside the render functions rather than through separate assets.

## Developer workflow
- Install dependencies with `pip install -r requirements.txt` into the workspace `.venv`.
- Preferred local run path is the VS Code task `Streamlit: Run App`; it launches `.venv\Scripts\python.exe -m streamlit run rf_analyzer\main.py`.
- There is no discoverable automated test suite or CI configuration in the repo today. After changes, validate by launching the Streamlit app and exercising the relevant controls.
- Because this is a rerun-driven Streamlit app, UI edits are best verified by changing a control and watching the downstream plots/metrics update instead of expecting persistent in-memory state.

## When adding features
- Keep new RF controls synchronized across all three layers: session-state defaults, sidebar/main controls, and the processing pipeline.
- If you add a new stage or change stage ordering, update both the producer in `main_window.py` and the fixed four-column consumers in `plots_panel.py`.
- If you extract logic from `main_window.py` into the currently sparse `signal/`, `metrics/`, or `impairments/` packages, move the call sites too; adding helper files alone will not change behavior.
- Use NumPy arrays throughout the signal path and keep complex samples in `np.complex128` form where the existing pipeline does.
