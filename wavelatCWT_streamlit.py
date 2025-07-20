import streamlit as st
import pandas as pd
import numpy as np
import pywt, mne, os
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import MinMaxScaler
import zipfile

st.title("EEG CWT Feature Extraction")
st.markdown("""
- Upload a ZIP file containing EEG CSVs (organized by class folders)
- Each CSV should have signals from multiple channels
- Features are extracted per segment, channel, and frequency band
""")

uploaded_zip = st.file_uploader("Upload ZIP file", type=["zip"])
fs = st.number_input("Sampling Frequency (Hz)", min_value=1, value=128)
n_segments = st.number_input("Number of Segments", min_value=1, value=5)
wavelet_name = st.selectbox("Wavelet Name", ["morl", "db4", "db6", "sym5", "sym8", "coif5", "coif3"], index=0)
scales = np.arange(1, 65)

all_cwt_features = []
error_count = 0
single_channel_files = []

if uploaded_zip:
    try:
        with zipfile.ZipFile(uploaded_zip) as z:
            class_folders = set([
                os.path.dirname(f)
                for f in z.namelist()
                if f.lower().endswith('.csv') and not os.path.dirname(f).startswith('__MACOSX')
            ])
            st.write(f"Detected emotion/class folders: {class_folders}")
            # Progress bar setup
            total_files = sum([
                len([
                    f for f in z.namelist()
                    if f.startswith(class_folder + '/')
                    and f.lower().endswith('.csv')
                    and not f.startswith('__MACOSX/')
                    and not os.path.basename(f).startswith('._')
                ]) for class_folder in class_folders
            ])
            progress = st.progress(0)
            processed_files = 0
            for class_folder in class_folders:
                class_label = os.path.basename(class_folder)
                csv_files = [
                    f for f in z.namelist()
                    if f.startswith(class_folder + '/')
                    and f.lower().endswith('.csv')
                    and not f.startswith('__MACOSX/')
                    and not os.path.basename(f).startswith('._')
                ]
                for csv_name in csv_files:
                    try:
                        with z.open(csv_name) as f:
                            df = pd.read_csv(f)
                            ch_names = df.columns.tolist()
                            data_columns = [col for col in ch_names if col.lower() not in ['label', 'target']]
                            if len(data_columns) < 2:
                                single_channel_files.append(csv_name)
                                processed_files += 1
                                progress.progress(processed_files / total_files)
                                continue
                            data = df[data_columns].values.T
                            info = mne.create_info(ch_names=data_columns, sfreq=fs, ch_types='eeg')
                            raw = mne.io.RawArray(data, info)
                            ica = mne.preprocessing.ICA(n_components=min(15, len(data_columns)), random_state=42, method='fastica')
                            ica.fit(raw)
                            sources = ica.get_sources(raw).get_data()
                            stds = np.std(sources, axis=1)
                            threshold = np.percentile(stds, 90)
                            ica.exclude = [i for i, s in enumerate(stds) if s > threshold]
                            raw_clean = ica.apply(raw.copy())
                            total_samples = raw_clean.n_times
                            seg_len = total_samples // n_segments
                            for seg_idx in range(n_segments):
                                start = seg_idx * seg_len
                                stop = start + seg_len
                                seg_data = raw_clean.get_data()[:, start:stop]
                                for ch_idx, ch_name in enumerate(data_columns):
                                    signal = seg_data[ch_idx]
                                    wavelet = pywt.ContinuousWavelet(wavelet_name)
                                    coefs, freqs = pywt.cwt(signal, scales, wavelet, 1.0 / fs)
                                    abs_coefs = np.abs(coefs)
                                    band_ranges = {
                                        "delta": (0.5, 4),
                                        "theta": (4, 8),
                                        "alpha": (8, 13),
                                        "beta":  (13, 30),
                                        "gamma": (30, 45)
                                    }
                                    for band, (low, high) in band_ranges.items():
                                        idxs = np.where((freqs >= low) & (freqs <= high))[0]
                                        band_coefs = abs_coefs[idxs, :].flatten()
                                        total_energy = np.sum(band_coefs ** 2)
                                        total_entropy = entropy(band_coefs / (np.sum(band_coefs) + 1e-12))
                                        coef_mean = np.mean(band_coefs)
                                        coef_std = np.std(band_coefs)
                                        coef_kurtosis = kurtosis(band_coefs)
                                        coef_skewness = skew(band_coefs)
                                        all_cwt_features.append({
                                            "class": class_label,
                                            "trial": os.path.splitext(os.path.basename(csv_name))[0],
                                            "segment": seg_idx + 1,
                                            "channel": ch_name,
                                            "band": band,
                                            "energy": total_energy,
                                            "entropy": total_entropy,
                                            "mean": coef_mean,
                                            "std": coef_std,
                                            "skewness": coef_skewness,
                                            "kurtosis": coef_kurtosis
                                        })
                        processed_files += 1
                        progress.progress(processed_files / total_files)
                    except Exception as e:
                        st.error(f"❌ Error on {csv_name}: {e}")
                        error_count += 1
                        processed_files += 1
                        progress.progress(processed_files / total_files)
                        continue
            if single_channel_files:
                st.warning(f"Skipped files (need >=2 channels):\n" + '\n'.join(single_channel_files))
    except zipfile.BadZipFile:
        st.error("Not a valid ZIP archive.")
    except Exception as e:
        st.error(f"ZIP extraction error: {e}")
    # Removed debug output for extracted features
    st.write(f"Total errors: {error_count}")
    if all_cwt_features:
        df_cwt = pd.DataFrame(all_cwt_features)
        # One-hot encode labels
        for class_label in df_cwt['class'].unique():
            df_cwt[f"label_{class_label.lower()}"] = (df_cwt["class"].str.lower() == class_label.lower()).astype(int)
        # Normalize
        feature_cols = ["energy", "entropy", "mean", "std", "skewness", "kurtosis"]
        scaler = MinMaxScaler()
        df_cwt[feature_cols] = scaler.fit_transform(df_cwt[feature_cols])
        st.write("All CWT features:", df_cwt)
        # Download band-wise CSVs
        bands = ["delta", "theta", "alpha", "beta", "gamma"]
        for band in bands:
            df_band = df_cwt[df_cwt['band'] == band]
            csv_data = df_band.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download {band} band features as CSV",
                data=csv_data,
                file_name=f"cwt_features_{band}.csv",
                mime="text/csv"
            )

        # MI Score computation and visualization
        st.header("Feature vs Class: Mutual Information Scores")
        selected_band = st.selectbox("Select Frequency Band for MI Analysis", bands)
        mi_df = None
        df_band = df_cwt[df_cwt['band'] == selected_band]
        if not df_band.empty:
            from sklearn.feature_selection import mutual_info_classif
            class_names = df_band['class'].unique()
            feature_cols_band = [col for col in feature_cols if col in df_band.columns]
            X = df_band[feature_cols_band].values
            mi_results = []
            for class_name in class_names:
                y_binary = (df_band['class'] == class_name).astype(int)
                mi_scores = mutual_info_classif(X, y_binary, discrete_features=False)
                for feat, score in zip(feature_cols_band, mi_scores):
                    mi_results.append({
                        "Class": class_name,
                        "Feature": feat,
                        "MI Score": score
                    })
            mi_df = pd.DataFrame(mi_results)
            # Pivot for grouped bar chart
            pivot_df = mi_df.pivot(index="Feature", columns="Class", values="MI Score")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_df.plot(kind="barh", ax=ax)
            ax.set_xlabel("Mutual Information Score")
            ax.set_title(f"MI Scores for {selected_band.upper()} Band vs All Classes")
            ax.legend(title="Class")
            st.pyplot(fig)
            st.write("MI Scores Table:", mi_df)
