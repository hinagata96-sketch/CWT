# EEG Feature Extraction Dashboard

This repository contains a unified Streamlit dashboard for extracting features from EEG data in four domains:
- Time Domain (TDMI)
- Frequency Domain (FDMI)
- Continuous Wavelet Transform (CWT)
- Discrete Wavelet Transform (DWT)

## How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the dashboard:
   ```bash
   streamlit run main_dashboard.py
   ```

## Project Structure
- `main_dashboard.py`: Main entry point for the dashboard
- `TDMI_app.py`, `fdmi_app.py`, `cwt_app.py`, `dwt_app.py`: Individual analysis modules
- `TDMI.py`, `FDMI.py`, `CWT.py`, `DWT.py`: Feature extraction logic
- `requirements.txt`: Python dependencies

## Usage
- Upload ZIP files containing EEG CSVs organized by class folders
- Select analysis domain from the sidebar
- Extract features, view results, and download CSVs

## License
MIT
